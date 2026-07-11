"""NL2SQL：两步路由（先选表 + 再生成SQL），支持大量表场景"""

import json
import re
import sqlite3

import pandas as pd
from openai import OpenAI
from loguru import logger

from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    DB_PATH, SQL_ALLOWED_OPS, SQL_BLOCKED_OPS, MAX_SQL_ROWS,
)


# 表选择 prompt
TABLE_SELECT_PROMPT = """你是财务数据库表选择专家。根据用户问题，从可用表清单中选择最相关的表。

**口径概念（重要）**：
- 文件名含"合并"的是【合并报表】（如"合并0：星辰集团"），反映整个集团/子集团合并后数据。
- 其他是【单体报表】（如"01深圳星辰数字科技集团股份有限公司"），反映单家公司数据。
- "深圳星辰数字科技集团股份有限公司"是集团的母公司/上市主体（单体口径），与"合并0：星辰集团"（合并口径）是两张不同的表。

**海外公司中英文对照（重要！用户常用中文，但表名是英文，必须映射）**：
- 香港星辰 / 香港公司 = "08STARNOVA (HONG KONG) LIMITED"
- 马来星辰 / 马来西亚星辰 / 马来公司 = "08-1STARNOVA MALAYSIA SDN. BHD."（有人民币、林吉特两个币种版本）
- 美国星辰 / 美国公司 = "10Starnova Graphics Inc"（有人民币、美元两个币种版本）
当用户用上述中文名提问时，必须选对应英文表名的表。多币种公司若用户**未指定币种，两个币种版本都要选**（如人民币版+美元版），让用户对照，不要只选一个。

规则：
1. 用户提到具体公司名 → 选该公司对应的表。
2. 用户提到"集团"或集团名（如"星辰集团"）但**未指明口径** → 同时选【母公司单体表】和【集团合并表】两张，让用户对照。
   例如问"星辰集团净利润" → 同时选 "t_01深圳星辰数字科技集团股份有限公司_..._利润表" 和 "t_合并0_星辰集团_..._利润表"。
3. 用户明确说"合并" → 只选合并表；明确说"母公司/单体/单家" → 只选单体表。
4. 同一报表类型在不同公司都相关时，可多选。
5. 仅返回JSON: {"tables": ["表名1", "表名2"], "reason": "选择理由"}"""

NL2SQL_PROMPT = """你是财务数据库SQL生成专家。根据用户问题生成SQLite查询。

规则：
1. 只生成 SELECT 查询
2. 列名含中文/空格/特殊字符时用双引号包裹，如 "期末数"
3. 表名也用双引号包裹，如 "t_xxx"
4. 金额字段是浮点数，可直接 SUM/AVG
5. 结果限制 {max_rows} 行以内
6. 涉及多表时使用 UNION ALL 或 JOIN
7. 如果列名是"列N"形式（无意义），跳过不用
8. 无法确定时返回 {{"error": "说明原因"}}

**关键：财务报表项目匹配规则**
- 财务报表中的"项目"、"科目"、"资产"、"负债"等文本列，其值通常带有中文数字前缀、缩进空格或后缀说明，
  例如"净利润"的实际值是 "五、净利润（净亏损以"-"号填列）"，"营业收入"是 "一、营业收入"。
- 因此对这类文本列做条件匹配时，**必须用 LIKE '%关键词%'，绝对不要用 = 精确匹配**。
- 同时为了返回上下文，SELECT 应包含项目列本身 + 数值列，并把项目列也返回给用户核对。
- 净利润优先匹配"归属于母公司所有者的净利润"或主"净利润"行，避免只取持续经营/终止经营分项。
- 示例：SELECT "项目","本期数" FROM "t_xxx" WHERE "项目" LIKE '%净利润%'

**多表查询（重要）**：
- 当查询涉及多张表（如同时查母公司单体表和合并表）时，**必须用 UNION ALL 合并**，
  并增加一列 "数据来源" 标识每行来自哪张表/哪个口径，例如：
  SELECT '深圳星辰(单体)' AS "数据来源", "项目", "本期数" FROM "t_01深圳星辰..._利润表" WHERE "项目" LIKE '%净利润%'
  UNION ALL
  SELECT '星辰集团(合并)' AS "数据来源", "项目", "本期数" FROM "t_合并0_星辰集团..._利润表" WHERE "项目" LIKE '%净利润%'
- "数据来源"标识要简洁标明公司简称 + 口径（单体/合并）+ 币种（若涉及多币种，如"美国星辰(单体/人民币)"、"美国星辰(单体/美元)"）。

**解读/推导/计算过程类问题（重要）**：
- 当用户问"怎么算来的"、"为什么"、"解读"、"分析"、"推导"时，**不要只查目标科目一行**，
  而要返回**整张相关报表的全部行**（不加 WHERE 项目过滤），让上层模型基于完整数据解释计算逻辑。
- 例如问"净利润2.69亿怎么算来的" → SQL: SELECT "项目","本期数","上年同期数" FROM "t_xxx_利润表" LIMIT 1000
  返回利润表所有项目（营业收入/成本/各项费用/利润），让上层逐步推导得出净利润。

**比率/百分比类问题（重要）**：
- 用户问"净利率/毛利率/费用率/同比增长率/占比"等比率类问题时，**严禁在SQL里写嵌套子查询除法**。
- **正确做法**：只 SELECT 出计算需要的两个原始数（如净利润和营业收入），由上层模型负责做除法。
- 示例（净利率）：
  ✅ 正确：SELECT "项目","本期数" FROM "t_xxx_利润表" WHERE "项目" LIKE '%净利润%' OR "项目" LIKE '%营业收入%'
  ❌ 错误：SELECT ROUND( (SELECT ... )/(SELECT ...) * 100, 2 ) AS "净利率"
- 嵌套子查询SQL极易出错（拼写/语法/类型转换），坚决避免。

返回JSON: {{"sql": "SELECT ...", "explanation": "查询说明"}}"""


# ============================================================
# 公司中英文别名映射：解决"中文提问 ↔ 英文表名"对不上的问题
# 海外公司的文件名是纯英文，同事用中文提问时需要这张表桥接
# 键=中文别名（用户可能的叫法）  值=文件名/表名里的英文关键词
# ============================================================
COMPANY_ALIASES = {
    "香港星辰": ["08STARNOVA", "STARNOVA", "HONG KONG"],
    "星辰香港公司": ["08STARNOVA", "STARNOVA", "HONG KONG"],
    "香港公司": ["08STARNOVA", "STARNOVA", "HONG KONG"],
    "马来星辰": ["08-1STARNOVA MALAYSIA", "MALAYSIA"],
    "马来西亚星辰": ["08-1STARNOVA MALAYSIA", "MALAYSIA"],
    "马来公司": ["08-1STARNOVA MALAYSIA", "MALAYSIA"],
    "马来西亚公司": ["08-1STARNOVA MALAYSIA", "MALAYSIA"],
    "美国星辰": ["10Starnova Graphics", "Starnova Graphics", "Graphics"],
    "美国公司": ["10Starnova Graphics", "Starnova Graphics", "Graphics"],
}


class SQLChain:
    def __init__(self):
        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    def generate_and_execute(self, question: str, table_info: list[dict],
                              on_step=None, pre_selected: list[dict] = None) -> dict:
        _step = on_step or (lambda *a, **kw: None)

        if not table_info:
            return {"success": False, "error": "数据库中暂无数据表"}

        # 步骤1：选表（如果上游已经选好，跳过本次LLM调用）
        if pre_selected:
            # 仍然走口径补全 + 币种补全（程序级，无LLM调用）
            selected_tables = self._augment_caliber(question, pre_selected, table_info)
            selected_tables = self._augment_currency(question, selected_tables, table_info)
            logger.info(f"跳过选表LLM，使用预选 + 补全: {len(selected_tables)}张")
        else:
            selected_tables = self._select_tables(question, table_info)

        if not selected_tables:
            return {"success": False, "error": "未找到与问题相关的数据表，请尝试更明确的问题"}

        logger.info(f"已选表 ({len(selected_tables)}): {[t['table_name'] for t in selected_tables][:5]}")
        _step("generate_sql", f"已匹配 {len(selected_tables)} 张表，正在生成SQL")

        # 步骤2：基于选中表的详细 schema 生成 SQL
        schema_desc = self._build_detailed_schema(selected_tables)

        # 强制约束：每张选中的表都必须在SQL中出现
        must_use_clause = ""
        if len(selected_tables) > 1:
            must_use_clause = (
                f"\n\n【强制要求】共有 {len(selected_tables)} 张表必须**全部**出现在 SQL 中，"
                f"用 UNION ALL 合并，并为每个分支加 '数据来源' 列标注口径：\n"
                + "\n".join(
                    f"  - {t['table_name']} "
                    f"（{'合并' if '合并' in t['source_file'] else '单体'}口径，"
                    f"标签建议: {self._suggest_source_label(t)}）"
                    for t in selected_tables
                )
            )

        response = self.client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": NL2SQL_PROMPT.format(max_rows=MAX_SQL_ROWS)},
                {"role": "user", "content": f"数据库表结构：\n{schema_desc}{must_use_clause}\n\n用户问题：{question}"},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)

        if "error" in result:
            return {"success": False, "error": result["error"]}

        sql = result.get("sql", "")
        explanation = result.get("explanation", "")

        # 自动修复：拼写错误 + UNION ALL 分支内的 LIMIT 移到末尾
        sql = self._fix_typos(sql)
        sql = self._fix_union_limit(sql)

        safety_check = self._validate_sql(sql)
        if not safety_check["safe"]:
            return {"success": False, "error": safety_check["reason"]}

        _step("execute_sql", "正在执行SQL查询")
        try:
            # 只读连接：纵深防御，即便校验被绕过也无法写库
            uri = f"file:{DB_PATH}?mode=ro"
            with sqlite3.connect(uri, uri=True) as conn:
                df = pd.read_sql_query(sql, conn)
            logger.info(f"SQL执行成功: {sql[:100]}... → {len(df)}行")
            return {
                "success": True,
                "sql": sql,
                "explanation": explanation,
                "data": df,
                "row_count": len(df),
                "selected_tables": [t["table_name"] for t in selected_tables],
            }
        except Exception as e:
            logger.error(f"SQL执行失败: {sql} → {e}")
            return {"success": False, "sql": sql, "error": str(e)}

    def _select_tables(self, question: str, table_info: list[dict]) -> list[dict]:
        """步骤1：先用关键词预筛 + LLM精选"""
        # 1.1 关键词预筛（基于问题中的报表类型）
        pre_filtered = self._keyword_prefilter(question, table_info)

        # 如果预筛后还是太多，做按公司+报表类型聚合的简短清单
        if len(pre_filtered) > 30:
            short_list = self._build_short_table_list(pre_filtered)
        else:
            short_list = "\n".join(
                f"- {t['table_name']} | 来源:{t['source_file']} | sheet:{t.get('sheet_name','')} | 行数:{t['row_count']}"
                for t in pre_filtered
            )

        # 1.2 LLM 选表
        try:
            response = self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": TABLE_SELECT_PROMPT},
                    {"role": "user", "content": f"可用表清单 ({len(pre_filtered)}张):\n{short_list}\n\n用户问题：{question}"},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            sel = json.loads(response.choices[0].message.content)
            sel_names = set(sel.get("tables", []))
            selected = [t for t in table_info if t["table_name"] in sel_names]

            # 程序级口径补全 + 币种补全：用户没明确指定时自动补齐单体↔合并、多币种
            selected = self._augment_caliber(question, selected, table_info)
            selected = self._augment_currency(question, selected, table_info)
            return selected
        except Exception as e:
            logger.warning(f"LLM选表失败: {e}, 使用关键词预筛结果")
            return pre_filtered[:5]

    def _fix_typos(self, sql: str) -> str:
        """修正LLM常见的SQL关键字拼写错误"""
        import re as _re
        # 关键字拼写修正（大小写不敏感，仅替换单词边界）
        typo_map = {
            r"\bSEST\b": "SELECT",
            r"\bSELCT\b": "SELECT",
            r"\bSELET\b": "SELECT",
            r"\bFORM\b": "FROM",
            r"\bWHRE\b": "WHERE",
            r"\bWEHRE\b": "WHERE",
            r"\bGROUO\s+BY\b": "GROUP BY",
            r"\bORDR\s+BY\b": "ORDER BY",
            r"\bUNINO\b": "UNION",
        }
        fixed = sql
        for pattern, correct in typo_map.items():
            new = _re.sub(pattern, correct, fixed, flags=_re.IGNORECASE)
            if new != fixed:
                logger.info(f"SQL拼写修正: {pattern} → {correct}")
                fixed = new
        return fixed

    def _fix_union_limit(self, sql: str) -> str:
        """修复 UNION ALL 中分支LIMIT导致的语法错误"""
        if " UNION ALL " not in sql.upper():
            return sql
        import re as _re
        matches = list(_re.finditer(r"\bLIMIT\s+(\d+)", sql, _re.IGNORECASE))
        if len(matches) < 2:
            return sql
        # 全部移除后在语句末尾统一加一个 LIMIT（取最后一个的值）：
        # 仅保留"最后一个"并不安全——它可能位于末分支子查询里而非语句末尾。
        # 删除必须从后往前，否则前面的删除会使后面 match 的偏移失效
        last_val = matches[-1].group(1)
        new_sql = sql
        for m in reversed(matches):
            new_sql = new_sql[:m.start()] + new_sql[m.end():]
        new_sql = _re.sub(r"\s+", " ", new_sql).strip().rstrip(";").strip()
        new_sql = f"{new_sql} LIMIT {last_val}"
        logger.info(f"SQL自动修复(LIMIT位置): {new_sql[:120]}...")
        return new_sql

    def _suggest_source_label(self, t: dict) -> str:
        """为表生成简短的'数据来源'标签：公司简称(单体/合并)"""
        fname = t["source_file"]
        is_merged = "合并" in fname
        # 提取公司简称：去扩展名 + 编号前缀（01/02-1/02-1-1/合并1-1：等）+ TB标记
        short = fname.replace(".xlsx", "").replace(".xls", "")
        short = re.sub(r"^合并\d+[-\d]*[：:]?", "", short)   # 合并0：/合并1-1： 等
        short = re.sub(r"^\d+[-\d]*", "", short)              # 01 / 02-1 / 02-1-1 等
        short = re.sub(r"_20\d{2}TB", "", short)              # _2025TB / _2026TB
        short = short.strip("：:_").strip()
        if len(short) > 12:
            short = short[:12]
        return f"{short}({'合并' if is_merged else '单体'})"

    def _augment_caliber(self, question: str, selected: list[dict],
                         all_tables: list[dict]) -> list[dict]:
        """口径自动补全：用户没明确指定口径时，若LLM只选了单体或合并的一边，自动补齐另一边。

        识别规则：
        - 用户**明确**说"合并/单体/母公司/单家"时，尊重用户意图，不补
        - 否则，对每张被选中的表，若是合并表则补对应母公司单体表，反之亦然
        """
        if not selected:
            return selected

        # 用户明确指明口径 → 不补
        explicit_consolidated = any(k in question for k in ["合并报表", "合并口径", "合并"])
        explicit_standalone = any(k in question for k in ["母公司", "单体", "单家", "本部", "单独"])
        if explicit_consolidated or explicit_standalone:
            return selected

        selected_names = {t["table_name"] for t in selected}
        augmented = list(selected)

        for t in selected:
            sheet = t.get("sheet_name", "")
            fname = t["source_file"]
            is_merged = "合并" in fname

            # 找对应口径的兄弟表
            for other in all_tables:
                if other["table_name"] in selected_names:
                    continue
                if other.get("sheet_name", "") != sheet:
                    continue
                other_is_merged = "合并" in other["source_file"]
                if is_merged == other_is_merged:
                    continue

                # 判断是否同一公司体系
                if self._same_company_group(fname, other["source_file"], question):
                    augmented.append(other)
                    selected_names.add(other["table_name"])
                    logger.info(
                        f"口径补全: {t['table_name']}（{'合并' if is_merged else '单体'}）"
                        f" → 追加 {other['table_name']}（{'合并' if other_is_merged else '单体'}）"
                    )

        return augmented

    # 文件名里可能出现的币种标识
    _CURRENCY_WORDS = ["人民币", "美元", "林吉特", "港币", "美金", "欧元", "日元",
                       "RMB", "USD", "MYR", "HKD", "EUR", "JPY"]

    def _strip_currency(self, fname: str) -> str:
        """去掉文件名里的币种括号，得到不含币种的基名"""
        pat = r"[（(][^）)]*(?:" + "|".join(self._CURRENCY_WORDS) + r")[^）)]*[）)]"
        return re.sub(pat, "", fname)

    def _augment_currency(self, question: str, selected: list[dict],
                          all_tables: list[dict]) -> list[dict]:
        """币种自动补全：同一公司同一报表有多个币种版本时，用户没指定币种就全选，让用户对照。

        - 用户明确说了某币种（如"美元"）→ 尊重用户，不补
        - 否则对每张被选中的多币种表，补齐同公司同sheet的其他币种版本
        """
        if not selected:
            return selected

        # 用户明确指定币种 → 不补
        if any(w in question for w in self._CURRENCY_WORDS + ["原币", "本位币"]):
            return selected

        selected_names = {t["table_name"] for t in selected}
        augmented = list(selected)

        for t in selected:
            base = self._strip_currency(t["source_file"])
            # 文件名本身不含币种标识 → 单币种文件，无需补
            if base == t["source_file"]:
                continue
            sheet = t.get("sheet_name", "")
            for other in all_tables:
                if other["table_name"] in selected_names:
                    continue
                if other.get("sheet_name", "") != sheet:
                    continue
                # 去币种后基名相同 = 同公司同报表的另一币种版本
                if self._strip_currency(other["source_file"]) == base:
                    augmented.append(other)
                    selected_names.add(other["table_name"])
                    logger.info(
                        f"币种补全: {t['table_name']} → 追加 {other['table_name']}"
                    )
        return augmented

    # 公司单体↔合并精确配对表：每条 = (问题关键词列表, 单体文件特征, 合并文件特征)
    # 精确闭环匹配：用户问A公司时，只补A公司自己的另一口径，不会误补到兄弟公司
    _COMPANY_PAIRS = [
        # 星辰集团母公司（上市主体）
        (["星辰集团", "星辰数字", "深圳星辰"], ["01深圳星辰"], ["合并0", "星辰集团"]),
        # 辰拓系
        (["辰拓"], ["02深圳市辰拓"], ["合并1：辰拓"]),
        (["星博"], ["02-1深圳星博"], ["合并1-1：星博"]),
        (["星源"], ["02-1-1广东星源"], []),  # 星源若无独立合并报表，不强补
        # 星美系
        (["星美"], ["03深圳星美"], ["合并2：星美"]),
        (["佳星"], ["03-1广东佳星"], []),
        # 海外
        (["香港星辰", "STARNOVA", "HONG KONG"], ["08STARNOVA"], ["合并3"]),
        (["马来星辰", "马来西亚", "STARNOVA MALAYSIA", "MALAYSIA"],
         ["08-1STARNOVA MALAYSIA"], ["合并3"]),
    ]

    def _same_company_group(self, fname_a: str, fname_b: str, question: str) -> bool:
        """判断两个文件是否为同一家公司的单体/合并配对（精确匹配，不跨公司）"""
        for q_keys, std_keys, mrg_keys in self._COMPANY_PAIRS:
            if not any(k in question for k in q_keys):
                continue
            a_std = any(k in fname_a for k in std_keys) if std_keys else False
            b_std = any(k in fname_b for k in std_keys) if std_keys else False
            a_mrg = any(k in fname_a for k in mrg_keys) if mrg_keys else False
            b_mrg = any(k in fname_b for k in mrg_keys) if mrg_keys else False
            if (a_std and b_mrg) or (a_mrg and b_std):
                return True
        return False

    def _keyword_prefilter(self, question: str, table_info: list[dict]) -> list[dict]:
        """根据问题关键词预筛表"""
        # 报表类型关键词
        sheet_keywords = {
            "资产负债": ["资产负债表"],
            "利润": ["利润表", "损益表"],
            "现金流": ["现金流量表"],
            "应收": ["应收"],
            "应付": ["应付"],
            "存货": ["存货", "库存"],
            "费用": ["费用", "管理费用", "销售费用", "研发费用"],
            "收入": ["收入", "营业收入"],
            "成本": ["成本"],
            "TB": ["TB", "试算"],
            "附注": ["附注"],
        }

        # 提取问题中的报表类型
        relevant_sheets = set()
        for q_kw, sheet_kws in sheet_keywords.items():
            if q_kw in question:
                relevant_sheets.update(sheet_kws)

        # 公司关键词匹配（支持中英文双向）
        # ① 先看问题里有没有海外公司的中文别名，翻译成文件名英文关键词
        alias_keys = []
        for alias, file_keys in COMPANY_ALIASES.items():
            if alias in question:
                alias_keys.extend(file_keys)

        company_filters = []
        if alias_keys:
            # 命中海外公司中文别名 → 精确按英文关键词匹配，避免"星辰"泛化到所有星辰公司
            for t in table_info:
                if any(k in t["source_file"] for k in alias_keys):
                    company_filters.append(t["table_name"])
        else:
            # ② 国内公司：文件名本身含中文，直接用中文片段匹配
            direct_hints = ["星博", "星美", "辰拓", "诚跃", "星锐", "辰华", "星辰软件",
                            "星源", "佳星", "东晟", "惠州", "江苏", "星辰", "合并", "集团"]
            for t in table_info:
                fname = t["source_file"]
                for hint in direct_hints:
                    if hint in fname and hint in question:
                        company_filters.append(t["table_name"])
                        break

        # 过滤掉低质量表（列名全是"列N"无意义的，如某些附注/明细表）
        def _is_low_quality(t):
            cols = t.get("columns", "")
            col_list = [c.strip() for c in cols.split(",") if c.strip()]
            if not col_list:
                return True
            # "列0/列1/..." 占比超过50% 即视为低质量
            noisy = sum(1 for c in col_list if c.startswith("列") and c[1:].isdigit())
            return noisy / len(col_list) >= 0.5

        # 用户没明确说"附注/明细"时，过滤掉这些次要表
        user_wants_附注 = "附注" in question or "明细" in question
        def _is_secondary(t):
            sn = t.get("sheet_name", "")
            return any(kw in sn for kw in ["附注", "明细", "Index", "调整分录",
                                            "Sheet2", "Sheet3", "xbase", "分析分录"])

        # 应用筛选
        filtered = []
        for t in table_info:
            sheet_match = not relevant_sheets or any(
                kw in t.get("sheet_name", "") for kw in relevant_sheets
            )
            company_match = not company_filters or t["table_name"] in company_filters

            # 跳过低质量列名表
            if _is_low_quality(t):
                continue
            # 用户未明确要求时跳过附注/明细
            if not user_wants_附注 and _is_secondary(t):
                continue

            if sheet_match and (not company_filters or company_match):
                filtered.append(t)

        # 如果筛得太严（0条），降级返回全部
        return filtered if filtered else table_info

    def _build_short_table_list(self, tables: list[dict]) -> str:
        """按公司+报表类型聚合，避免清单过长"""
        # 按 source_file 分组
        groups = {}
        for t in tables:
            groups.setdefault(t["source_file"], []).append(t)

        lines = []
        for fname, ts in groups.items():
            caliber = "合并报表" if "合并" in fname else "单体报表"
            lines.append(f"📁 {fname} [{caliber}]")
            for t in ts:
                lines.append(f"   - {t['table_name']} (sheet:{t.get('sheet_name','')}, 行数:{t['row_count']})")
        return "\n".join(lines)

    def _build_detailed_schema(self, tables: list[dict]) -> str:
        """为选中的表构建详细schema（含列名）"""
        lines = []
        for t in tables:
            cols = t.get("columns", "")
            lines.append(
                f"表名: \"{t['table_name']}\"\n"
                f"  来源: {t['source_file']} (sheet: {t.get('sheet_name','')})\n"
                f"  列: {cols}\n"
                f"  行数: {t['row_count']}"
            )
        return "\n\n".join(lines)

    def _validate_sql(self, sql: str) -> dict:
        if not sql or not sql.strip():
            return {"safe": False, "reason": "SQL为空"}

        sql_upper = sql.upper().strip()

        first_word = sql_upper.split()[0] if sql_upper.split() else ""
        if first_word not in SQL_ALLOWED_OPS:
            return {"safe": False, "reason": f"仅允许SELECT查询，检测到: {first_word}"}

        for op in SQL_BLOCKED_OPS:
            pattern = rf"\b{op}\b"
            if re.search(pattern, sql_upper):
                return {"safe": False, "reason": f"检测到危险操作: {op}"}

        if ";" in sql.strip().rstrip(";"):
            return {"safe": False, "reason": "禁止多语句执行"}

        # 拦截SQL注释符号：可藏匿危险操作或绕过上面的关键词检测
        if "--" in sql or "/*" in sql or "*/" in sql:
            return {"safe": False, "reason": "SQL含注释符号，已拦截"}

        return {"safe": True, "reason": ""}

"""问答编排器：统一调度意图路由 + 数据查询 + 文本RAG + 混合查询 + 澄清"""

from openai import OpenAI
from loguru import logger

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from query_router import QueryRouter
from sql_chain import SQLChain
from rag_chain import RAGChain
from table_extractor import TableExtractor
from image_store import ImageStore
from metrics_calculator import compute_metrics


class Orchestrator:
    def __init__(self):
        self.router = QueryRouter()
        self.sql_chain = SQLChain()
        self.rag_chain = RAGChain()
        self.table_extractor = TableExtractor()
        self.image_store = ImageStore()
        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    FOLLOWUP_HINTS = ["这个", "那个", "上面", "刚才", "刚刚", "前面", "之前",
                      "再", "继续", "也", "另外", "还有", "怎么来", "怎么算",
                      "为什么", "解释", "解读", "推导", "分析", "对比"]

    # 自包含判定：出现公司名 + 报表科目即视为问题完整，不必套用上文
    _COMPANY_WORDS = ["星辰", "辰拓", "星美", "星博", "佳星", "星源", "诚跃", "星锐",
                      "辰华", "东晟", "惠州", "江苏", "合并", "集团",
                      "STARNOVA", "MALAYSIA", "Graphics", "香港", "马来", "美国"]
    _METRIC_WORDS = ["净利润", "营业收入", "营业成本", "利润总额", "营业利润",
                     "资产", "负债", "权益", "毛利", "费用", "现金流",
                     "净利率", "毛利率", "收入", "成本", "税"]

    def _is_followup(self, question: str) -> bool:
        """启发式判断是否需要参考上文。

        修复：原来只要 len<15 就判为追问，会把"星美净利润"这类**已含公司名+科目**
        的完整短问题也强制套上一轮上下文改写，导致串到别家公司/别的科目。
        现在：公司名 + 科目俱全 → 自包含，优先短路不套上文（即使含"对比/分析/解释"
        等提示词也不例外，否则"对比星辰和辰拓净利率"仍会被改写串公司）；否则
        再按提示词/短问题判追问。"""
        q = question.strip()
        has_company = any(k in q for k in self._COMPANY_WORDS)
        has_metric = any(k in q for k in self._METRIC_WORDS)
        if has_company and has_metric:
            return False
        if any(k in q for k in self.FOLLOWUP_HINTS):
            return True
        return len(q) < 15

    # 上下文管理参数：DeepSeek上下文64K tokens。
    # 中文约 1 token ≈ 1.5 字符，历史预算 4 万字符 ≈ 2.7万 tokens，
    # 给系统提示词/表结构/查询结果留足余量（旧值12万字符≈8万tokens会撑爆上下文）
    MAX_HISTORY_CHARS = 40_000

    def _build_history_context(self, chat_history: list) -> str:
        """构建完整对话历史。从最新往前累加，超出预算时丢弃最早的。"""
        if not chat_history:
            return ""

        msgs = []
        total = 0
        # 从最新往前遍历
        for msg in reversed(chat_history):
            role = "用户" if msg["role"] == "user" else "助手"
            content = msg.get("content") or ""
            line = f"{role}: {content}"
            if total + len(line) > self.MAX_HISTORY_CHARS:
                logger.info(f"历史超长，从第{len(chat_history)-len(msgs)}条之前的丢弃")
                break
            msgs.append(line)
            total += len(line) + 1

        # 反转回时间顺序
        msgs.reverse()
        return "\n".join(msgs)

    def _resolve_with_context(self, question: str, chat_history: list) -> str:
        """如果是追问，用LLM把它改写为独立完整问题"""
        if not chat_history or not self._is_followup(question):
            return question

        context = self._build_history_context(chat_history)

        try:
            resp = self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content":
                        "你的任务是把用户最新的问题补全为独立完整的财务查询问题。\n"
                        "结合上下文，把代词（这个/那个/上面）替换为具体的公司名、报表类型、口径、年份和科目。\n"
                        "如果原问题已经完整或没有上下文可用，原样返回。\n"
                        "只返回JSON: {\"question\": \"...\"}"
                    },
                    {"role": "user", "content": f"对话历史：\n{context}\n\n用户当前问题：{question}\n请返回补全后的完整问题。"},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            import json
            new_q = json.loads(resp.choices[0].message.content).get("question", question)
            if new_q and new_q.strip() and new_q != question:
                logger.info(f"上下文补全: '{question}' → '{new_q}'")
                return new_q
        except Exception as e:
            logger.warning(f"上下文补全失败: {e}")
        return question

    def ask(self, question: str, chat_history: list = None,
            on_step=None, stream: bool = False) -> dict:
        """
        on_step(stage_key, message): 阶段进度回调
            stage_key: route / context / select_table / generate_sql / execute_sql / summarize
        stream: 是否流式返回最终解读（data类型时返回'answer_stream'生成器）
        """
        _step = on_step or (lambda *a, **kw: None)

        # ① 优化：第一轮（无历史）或不含追问词的短问题，跳过上下文补全LLM调用
        if chat_history and self._is_followup(question):
            _step("context", "正在分析上下文")
            resolved_q = self._resolve_with_context(question, chat_history)
        else:
            resolved_q = question

        # ② 优化：意图路由 + 选表合并为一次 LLM 调用
        _step("route", "正在识别意图并匹配相关表")
        table_info = self.table_extractor.get_all_tables_info()

        # 构造候选表清单（精简：仅传公司+sheet名，不传完整列）
        if table_info:
            # 按文件分组聚合
            groups = {}
            for t in table_info:
                groups.setdefault(t["source_file"], []).append(t.get("sheet_name", ""))
            table_list_str = "\n".join(
                f"📁 {fname} [{'合并' if '合并' in fname else '单体'}]: {', '.join(sheets[:25])}"
                for fname, sheets in groups.items()
            )
        else:
            table_list_str = ""

        # 文本文档清单一并给路由器：生活化提问（如"12点能去吃饭吗"）
        # 需要靠文档清单（考勤制度等）才能正确归为 text 而不是 clarify
        try:
            text_docs = self.rag_chain.vector_store.get_source_files()
        except Exception:
            text_docs = []
        route_result = self.router.classify_and_select(
            resolved_q, table_list_str, table_count=len(table_info),
            text_docs=text_docs,
        )
        intent = route_result["intent"]
        pre_selected_table_names = route_result.get("tables", [])

        # 把表名字符串转回完整 table_info（用于后续 SQL 生成）
        info_map = {t["table_name"]: t for t in table_info}
        pre_selected = [info_map[n] for n in pre_selected_table_names if n in info_map]

        logger.info(f"编排器: 问题='{resolved_q[:50]}...' 意图={intent} 预选{len(pre_selected)}张表")

        if intent == "data":
            return self._handle_data(resolved_q, table_info, _step, stream, pre_selected)
        elif intent == "text":
            return self._handle_text(resolved_q, _step, stream)
        elif intent == "hybrid":
            return self._handle_hybrid(resolved_q, table_info, _step, stream, pre_selected)
        else:
            return self._handle_clarify(resolved_q)

    def _handle_data(self, question: str, table_info: list[dict],
                     on_step=None, stream: bool = False,
                     pre_selected: list[dict] = None) -> dict:
        _step = on_step or (lambda *a, **kw: None)

        if not table_info:
            return {
                "type": "data",
                "answer": "暂无结构化数据表。请先上传包含表格的文档（Excel/CSV/PDF）。",
                "data": None,
                "sources": [],
            }

        result = self.sql_chain.generate_and_execute(
            question, table_info, on_step=_step, pre_selected=pre_selected,
        )

        if result["success"]:
            _step("summarize", "正在生成解读")
            summary = self._summarize_data(
                question, result["data"], result["explanation"],
                result.get("selected_tables", []),
                stream=stream,
            )
            return {
                "type": "data",
                "answer": summary,
                "sql": result["sql"],
                "data": result["data"],
                "row_count": result["row_count"],
                "sources": self._build_table_sources(
                    result.get("selected_tables", []), table_info
                ),
            }
        else:
            return {
                "type": "data",
                "answer": f"数据查询失败：{result['error']}",
                "data": None,
                "sources": [],
            }

    def _build_table_sources(self, selected_tables: list, table_info: list[dict]) -> list[dict]:
        """把选中的表映射为可预览/下载的来源（文件+sheet+口径）"""
        info_map = {t["table_name"]: t for t in table_info}
        sources = []
        seen = set()
        for tname in selected_tables:
            meta = info_map.get(tname)
            if not meta:
                continue
            key = (meta["source_file"], meta.get("sheet_name", ""))
            if key in seen:
                continue
            seen.add(key)
            caliber = "合并" if "合并" in meta["source_file"] else "单体"
            sources.append({
                "type": "table_source",
                "file": meta["source_file"],
                "sheet": meta.get("sheet_name", ""),
                "caliber": caliber,
                "table_name": tname,
            })
        return sources

    def _handle_text(self, question: str, on_step=None, stream: bool = False) -> dict:
        _step = on_step or (lambda *a, **kw: None)
        _step("select_table", "正在检索文档片段")
        _step("summarize", "正在生成回答")
        result = self.rag_chain.query(question)
        # 截图只取相关度最高来源文件的，避免多文件几十张混杂轰炸
        sources = result.get("sources", [])
        img_sources = sources
        if sources:
            top_file = sources[0]["file"]
            img_sources = [s for s in sources if s["file"] == top_file]
        page_images = self.image_store.get_images_for_sources(img_sources)
        return {
            "type": "text",
            "answer": result["answer"],
            "data": None,
            "sources": result["sources"],
            "snippets": result.get("snippets", []),
            "page_images": page_images,
        }

    def _handle_hybrid(self, question: str, table_info: list[dict],
                       on_step=None, stream: bool = False,
                       pre_selected: list[dict] = None) -> dict:
        _step = on_step or (lambda *a, **kw: None)
        data_result = self._handle_data(question, table_info, on_step=_step,
                                         pre_selected=pre_selected)
        text_result = self._handle_text(question, on_step=_step)

        combined_context = ""
        if data_result.get("data") is not None and not data_result["data"].empty:
            combined_context += f"【数据查询结果】\n{self._df_preview(data_result['data'], 100)}\n\n"
            # 附带已带口径标注的数据解读，避免合成回答时混淆单体/合并口径
            if data_result.get("answer"):
                combined_context += f"【数据口径说明】\n{data_result['answer']}\n\n"
        if text_result.get("answer"):
            combined_context += f"【文本检索结果】\n{text_result['answer']}\n\n"

        response = self.client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "你是财务问答助手。请综合数据查询结果和文本检索结果，给出完整回答。使用简体中文。"},
                {"role": "user", "content": f"综合信息：\n{combined_context}\n用户问题：{question}"},
            ],
            temperature=0.1,
        )

        all_sources = text_result.get("sources", [])
        if data_result.get("sources"):
            all_sources.extend(data_result["sources"])

        return {
            "type": "hybrid",
            "answer": response.choices[0].message.content,
            "data": data_result.get("data"),
            "sources": all_sources,
        }

    def _handle_clarify(self, question: str) -> dict:
        table_info = self.table_extractor.get_all_tables_info()
        stats = self.rag_chain.vector_store.get_stats()

        hints = []
        if table_info:
            hints.append(f"数据表: {', '.join(t['table_name'] for t in table_info[:5])}")
        if stats["total_chunks"] > 0:
            hints.append(f"文本知识库: {stats['total_chunks']}个文本块")

        available = "\n".join(hints) if hints else "暂无文档"

        return {
            "type": "clarify",
            "answer": f"您的问题不够明确，请补充细节。\n\n当前知识库包含：\n{available}\n\n请尝试更具体的提问，例如：\n- 数据类：\"2024年管理费用合计是多少\"\n- 制度类：\"差旅费报销标准是什么\"",
            "data": None,
            "sources": [],
        }

    @staticmethod
    def _df_preview(df, rows: int = 30) -> str:
        """DataFrame转文本给LLM看。必须显式指定float格式：
        同列多量级数值时 pandas 默认切科学计数法（2.922935e+07），
        LLM 拿到的数字只剩7位有效数字，会导致回答尾数错误。"""
        return df.head(rows).to_string(
            index=False, float_format=lambda x: f"{x:,.2f}"
        )

    EXPLAIN_KEYWORDS = ["怎么算", "如何计算", "怎么计算", "怎么得", "怎么来",
                        "解读", "解释", "推导", "分析", "为什么", "原因",
                        "构成", "组成", "明细", "细节"]

    def _is_explain_question(self, question: str) -> bool:
        return any(k in question for k in self.EXPLAIN_KEYWORDS)

    def _summarize_data(self, question: str, df, explanation: str,
                        selected_tables: list = None, stream: bool = False):
        is_explain = self._is_explain_question(question)
        # 解读类问题给更多行，确保模型有完整上下文
        preview_rows = 100 if is_explain else 30
        data_preview = self._df_preview(df, preview_rows)

        table_note = ""
        if selected_tables:
            calibers = []
            for t in selected_tables:
                cal = "合并" if "合并" in t else "单体"
                calibers.append(f"{t}（{cal}口径）")
            table_note = f"\n本次数据来自以下表：\n" + "\n".join(calibers)

        base_rules = (
            "你是资深财务数据分析助手。根据查询结果用自然语言回答问题，数字必须严格来自查询结果。使用简体中文。\n"
            "通用要求：\n"
            "0. **结论先行**：第一句直接给出核心结论（判断类先答是/否，数值类先给数字），"
            "口径说明、对比细节、依据放在结论之后。\n"
            "1. 必须**明确标注每个数字来自哪张表/哪个口径（单体或合并）**，避免用户误读。\n"
            "2. 如果结果中含'数据来源'列，必须按不同来源分别列出对应数字。\n"
            "3. 同时有单体和合并数据时分别说明，并提示口径差异（合并已抵消内部交易）。\n"
            "4. 金额用千分位逗号展示，标注单位（元）。**数字必须逐位照抄查询结果，"
            "包括小数点后两位，严禁四舍五入、取整或改写任何一位**（如查询结果是12345678.99，"
            "必须写12,345,678.99，不得写12,345,680.00或约1235万）。概算表述（约XX亿元）只能"
            "作为精确数字之后的补充说明。\n"
            "5. 严禁编造查询结果中没有的数字。\n"
            "6. **判断数据归属哪家公司，一律以表名/文件名为准**；查询结果里若出现'编制单位'字段，"
            "可能因制表笔误而填错（如美国公司表里编制单位误写成马来公司），**不得据此改判公司归属，也不要向用户复述这个可能错误的编制单位**。\n"
            "7. **本期数与上年同期数**：报表表名含年份N时，'本期数/本年数/本年累计'= N年数据，'上年同期数/上期数'= N-1年数据。"
            "例如2025年报表的上年同期数就是2024年数据，2024年报表的上年同期数就是2023年数据。"
            "用户问某年数据时，必须根据报表年份判断该数据在哪一列，**如果上年同期数列有值就直接引用，不得说'未提供数据'**。"
        )

        # 比率/同比的算术由 Python 精确计算（确定性逻辑不交给概率模型），算出后作为权威事实注入
        metrics_block = compute_metrics(df, question)
        if metrics_block:
            base_rules += (
                "\n8. **比率、同比与变动已由系统精确计算**：下方'查询结果'后附有【系统精确计算结果】，"
                "其中的比率(%)、同比、以及'变动 X 个百分点'都是程序按公式算出的，"
                "**必须原样引用这些数值，严禁自己重新做除法/减法或改动小数位**；"
                "若某项系统未给出，才可基于原始数字自行计算并注明。"
            )

        if is_explain:
            extra = (
                "\n\n【解读/推导模式】用户在追问计算过程或解读原因，请：\n"
                "A. 按报表的标准会计公式逐步推导（如：营业利润 = 营业收入 − 营业成本 − 税金及附加 − 各项费用 + 其他收益 + 投资收益 ± 公允价值变动 − 减值损失 + 资产处置；利润总额 = 营业利润 + 营业外收入 − 营业外支出；净利润 = 利润总额 − 所得税）。\n"
                "B. 列出关键科目的数字+公式，分行展示推导链路，**最终得出的数字必须与查询结果中目标科目的金额完全一致**。\n"
                "C. 如果差额对不上，说明可能存在四舍五入差异或科目归类差异，**不要硬凑**。\n"
                "D. 末尾给出业务层面的简要解读（毛利率、费用占比、同比变动等），但所有比率必须基于查询结果中的数字算出。"
            )
            system_prompt = base_rules + extra
        else:
            system_prompt = base_rules

        metrics_note = f"\n\n{metrics_block}" if metrics_block else ""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"查询说明：{explanation}{table_note}\n\n查询结果：\n{data_preview}\n共{len(df)}行{metrics_note}\n\n用户问题：{question}"},
        ]

        if stream:
            # 流式：返回生成器，逐 chunk yield
            def _gen():
                resp = self.client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=messages,
                    temperature=0,
                    stream=True,
                )
                for chunk in resp:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        yield delta
            return _gen()

        response = self.client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content

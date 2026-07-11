"""财务比率精确计算：把比率/同比类算术从 LLM 收回到 Python，保证 100% 准确。

设计原则（守底线）：
- 只在能**高置信度**定位到所需数字时才计算，否则跳过该指标，回退给 LLM 原有行为，绝不弄巧成拙。
- 若结果含"数据来源"列（UNION ALL 多口径/多币种），按来源分组分别计算，避免跨口径混算。
- 算出的结果作为"权威事实"注入解读 prompt，要求 LLM 直接引用、不得重算。
"""

import re

import pandas as pd

# 数值列名候选（按优先级匹配）
_CURRENT_COL_KEYS = ["本期数", "本期金额", "本年累计", "本年数", "本期", "期末数", "期末余额", "年末数"]
_PRIOR_COL_KEYS = ["上年同期数", "上年同期", "上期数", "上年数", "上年累计", "年初数", "年初余额"]

# 文本项目列识别用的财务科目特征词
_ITEM_HINT_WORDS = ["营业", "利润", "收入", "成本", "费用", "资产", "负债", "权益", "现金"]

# 比率定义：name → (分子科目关键词列表, 分母科目关键词, 说明)
# 分子可由多个科目相加（如期间费用）
_RATIO_DEFS = {
    "毛利率": {"trigger": ["毛利率", "毛利"], "kind": "gross"},
    # "利润率"为常见口语，默认按净利率（净利润/营业收入）处理，与财务惯例一致
    "净利率": {"trigger": ["净利率", "销售净利率", "净利润率", "利润率"], "num": ["净利润"], "den": "营业收入"},
    "营业利润率": {"trigger": ["营业利润率"], "num": ["营业利润"], "den": "营业收入"},
    "期间费用率": {"trigger": ["期间费用率", "费用率"],
               "num": ["销售费用", "管理费用", "研发费用", "财务费用"], "den": "营业收入"},
    "销售费用率": {"trigger": ["销售费用率"], "num": ["销售费用"], "den": "营业收入"},
    "管理费用率": {"trigger": ["管理费用率"], "num": ["管理费用"], "den": "营业收入"},
    "研发费用率": {"trigger": ["研发费用率"], "num": ["研发费用"], "den": "营业收入"},
    "财务费用率": {"trigger": ["财务费用率"], "num": ["财务费用"], "den": "营业收入"},
}

# 同比可针对的科目：display → 匹配关键词
_YOY_ITEMS = {
    "营业收入": "营业收入", "营业成本": "营业成本", "净利润": "净利润",
    "营业利润": "营业利润", "利润总额": "利润总额",
    "销售费用": "销售费用", "管理费用": "管理费用",
    "研发费用": "研发费用", "财务费用": "财务费用",
}
# 注意：仅列"同比"(比上年同期)触发词。"环比"(比上期/上季)是不同口径，系统只有
# 上年同期列(prior_col)、无上期列，若纳入会把同比数据误标为环比，故排除。
_YOY_TRIGGERS = ["同比", "增长率", "增幅", "增长了", "较上年", "较去年", "比上年", "比去年",
                 "增减", "相比", "提升", "下降"]


def _num(v):
    """尽力把单元格转成 float，无法转换返回 None"""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).replace(",", "").replace("，", "").strip()
    s = re.sub(r"[元¥￥$%\s]", "", s)
    if s in ("", "-", "—", "None", "nan", "NaN"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _fmt(v):
    return f"{v:,.2f}"


def _find_item_col(df):
    """找到'项目/科目'文本列：取含财务科目特征词最多的列"""
    best, best_score = None, 0
    for col in df.columns:
        score = 0
        for v in df[col].head(60):
            s = str(v)
            if any(w in s for w in _ITEM_HINT_WORDS):
                score += 1
        if score > best_score:
            best, best_score = col, score
    return best if best_score >= 2 else None


def _find_value_cols(df):
    """返回 (本期列名, 上年同期列名)，找不到为 None"""
    cur = prior = None
    for col in df.columns:
        cn = str(col)
        if cur is None and any(k in cn for k in _CURRENT_COL_KEYS):
            cur = col
        if prior is None and any(k in cn for k in _PRIOR_COL_KEYS):
            prior = col
    return cur, prior


def _find_source_col(df):
    """多口径 UNION ALL 时的'数据来源'列"""
    for col in df.columns:
        if "数据来源" in str(col) or "来源" == str(col):
            return col
    return None


def _extract_item(sub, item_col, value_col, keyword):
    """从子表里按关键词匹配科目行，取数值列的值；净利润做特殊优选"""
    rows = sub[sub[item_col].astype(str).str.contains(keyword, na=False, regex=False)]
    if rows.empty:
        return None
    if keyword == "净利润":
        # 优先归母净利润 → 排除持续/终止经营/少数股东等分项 → 兜底第一行
        prefer = rows[rows[item_col].astype(str).str.contains("归属于母公司", na=False)]
        if not prefer.empty:
            rows = prefer
        else:
            main = rows[~rows[item_col].astype(str).str.contains(
                "持续经营|终止经营|少数股东|调整", na=False, regex=True)]
            if not main.empty:
                rows = main
    return _num(rows.iloc[0][value_col])


def _calc_ratio(sub, item_col, value_col, d):
    """按比率定义 d 计算指定数值列(本期或上年同期)的比率。
    返回 (百分比值float, 算式明细str)；定位不到所需科目或分母为0时返回 None（静默回退）。"""
    if value_col is None:
        return None
    if d.get("kind") == "gross":
        rev = _extract_item(sub, item_col, value_col, "营业收入")
        cost = _extract_item(sub, item_col, value_col, "营业成本")
        if rev and cost is not None and rev != 0:
            val = (rev - cost) / rev * 100
            detail = (f"(营业收入 {_fmt(rev)} − 营业成本 {_fmt(cost)}) "
                      f"/ 营业收入 {_fmt(rev)} × 100%")
            return val, detail
        return None
    parts = [_extract_item(sub, item_col, value_col, k) for k in d["num"]]
    den = _extract_item(sub, item_col, value_col, d["den"])
    if all(p is not None for p in parts) and den and den != 0:
        numer = sum(parts)
        if len(d["num"]) > 1:
            detail = " + ".join(f"{k} {_fmt(p)}" for k, p in zip(d["num"], parts))
            numer_str = f"({detail})"
        else:
            numer_str = f"{d['num'][0]} {_fmt(numer)}"
        val = numer / den * 100
        return val, f"{numer_str} / {d['den']} {_fmt(den)} × 100%"
    return None


def _compute_for_subset(sub, item_col, cur_col, prior_col, question, label=""):
    """对单一口径子表计算用户问到的比率与同比，返回事实行列表"""
    facts = []
    tag = f"（{label}）" if label else ""

    # —— 比率类（本期列；若有上年同期列则一并算上年比率与变动，全部落到 Python）——
    if cur_col:
        for name, d in _RATIO_DEFS.items():
            if not any(t in question for t in d["trigger"]):
                continue
            cur_res = _calc_ratio(sub, item_col, cur_col, d)
            if cur_res is None:
                continue
            cur_val, cur_detail = cur_res
            prior_res = _calc_ratio(sub, item_col, prior_col, d)
            if prior_res is not None:
                prior_val, prior_detail = prior_res
                cur_r, prior_r = round(cur_val, 2), round(prior_val, 2)
                facts.append(f"- {name}{tag}（本期）= {cur_detail} = {cur_r}%")
                facts.append(f"- {name}{tag}（上年同期）= {prior_detail} = {prior_r}%")
                facts.append(
                    f"- {name}{tag} 变动 = {cur_r}% − {prior_r}% "
                    f"= {round(cur_r - prior_r, 2)} 个百分点"
                )
            else:
                facts.append(f"- {name}{tag} = {cur_detail} = {round(cur_val, 2)}%")

    # —— 同比类（需本期 + 上年同期两列）——
    if cur_col and prior_col and any(t in question for t in _YOY_TRIGGERS):
        mentioned = [disp for disp, kw in _YOY_ITEMS.items() if disp in question]
        targets = mentioned or ["营业收入", "净利润"]
        for disp in targets:
            kw = _YOY_ITEMS[disp]
            cur = _extract_item(sub, item_col, cur_col, kw)
            pri = _extract_item(sub, item_col, prior_col, kw)
            if cur is not None and pri not in (None, 0):
                val = (cur - pri) / abs(pri) * 100
                facts.append(
                    f"- {disp}同比{tag} = (本期 {_fmt(cur)} − 上年同期 {_fmt(pri)}) "
                    f"/ |上年同期 {_fmt(pri)}| × 100% = {round(val, 2)}%"
                )
    return facts


def compute_metrics(df, question: str) -> str:
    """主入口：返回可注入 prompt 的'系统精确计算结果'文本块；无可算项返回空串。"""
    try:
        if df is None or df.empty or not question:
            return ""
        need_ratio = any(
            t in question for d in _RATIO_DEFS.values() for t in d["trigger"])
        need_yoy = any(t in question for t in _YOY_TRIGGERS)
        if not (need_ratio or need_yoy):
            return ""

        item_col = _find_item_col(df)
        cur_col, prior_col = _find_value_cols(df)
        if item_col is None or cur_col is None:
            return ""

        src_col = _find_source_col(df)
        all_facts = []
        if src_col:
            for src in df[src_col].dropna().unique():
                sub = df[df[src_col] == src]
                all_facts += _compute_for_subset(
                    sub, item_col, cur_col, prior_col, question, label=str(src))
        else:
            all_facts += _compute_for_subset(
                df, item_col, cur_col, prior_col, question)

        if not all_facts:
            return ""
        return (
            "【系统精确计算结果（已由程序按公式算出，请直接引用以下数值，"
            "不要自行重算，也不要改动小数）】\n" + "\n".join(all_facts)
        )
    except Exception:
        # 任何异常都静默回退，绝不影响主流程
        return ""

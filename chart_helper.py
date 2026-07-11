"""财务数据可视化：智能判断能否出图 + 推荐图型 + 交互渲染(plotly) + PNG下载(matplotlib中文)"""

import io
import re

import pandas as pd
import streamlit as st

# matplotlib 仅用于生成可下载的 PNG（复用项目中文字体，绝不乱码）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from config import BASE_DIR

# 注册中文字体
_FONT_PATH = BASE_DIR / "data" / "NotoSansSC-Regular.ttf"
if _FONT_PATH.exists():
    try:
        font_manager.fontManager.addfont(str(_FONT_PATH))
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(
            fname=str(_FONT_PATH)
        ).get_name()
    except Exception:
        pass
matplotlib.rcParams["axes.unicode_minus"] = False

_LABELS = {"line": "📈 折线图", "bar": "📊 柱状图", "pie": "🥧 饼图"}
_MAX_PLOT_ROWS = 50  # 超过则截断，避免图太挤


def _to_numeric(series: pd.Series) -> pd.Series:
    """把带千分位逗号/全角符号的列转成数值"""
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("，", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def _numeric_cols(df: pd.DataFrame) -> list:
    """识别数值列：至少一半的值能转成数字"""
    cols = []
    threshold = max(2, len(df) * 0.5)
    for c in df.columns:
        if _to_numeric(df[c]).notna().sum() >= threshold:
            cols.append(c)
    return cols


def suggest_chart(df: pd.DataFrame, question: str = "") -> dict:
    """判断这批数据是否适合可视化，并推荐图型。

    返回 {ok, recommended, cat_col, num_cols, cat_cols}
    不适合（单值/无数值列/无分类维度）时 ok=False，前端不显示图表入口。
    """
    if df is None or df.empty or len(df) < 2:
        return {"ok": False}

    num_cols = _numeric_cols(df)
    if not num_cols:
        return {"ok": False}

    cat_cols = [c for c in df.columns if c not in num_cols]
    if not cat_cols:
        return {"ok": False}

    # 按区分度（唯一值数量）降序：唯一值多的列更适合做X轴。
    # 避免选到"数据来源"这种只有单一值的列——否则多行会堆叠在一个柱上。
    cat_cols = sorted(cat_cols, key=lambda c: df[c].astype(str).nunique(), reverse=True)
    cat_col = cat_cols[0]
    q = question or ""

    # 推荐图型
    time_kw = any(k in q for k in ["趋势", "增长", "月度", "各月", "逐月", "环比",
                                    "同比", "历年", "季度", "走势", "变化"])
    cat_vals = df[cat_col].astype(str).tolist()[:6]
    time_like = any(re.search(r"\d{4}|月|季|Q[1-4]|年度|期", v) for v in cat_vals)
    pie_kw = any(k in q for k in ["占比", "构成", "结构", "比例", "分布", "组成", "明细"])

    if time_kw or time_like:
        rec = "line"
    elif pie_kw and len(df) <= 12:
        rec = "pie"
    else:
        rec = "bar"

    return {
        "ok": True,
        "recommended": rec,
        "cat_col": cat_col,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "row_count": len(df),
    }


def render_chart_section(df: pd.DataFrame, question: str, key_prefix: str):
    """在 Streamlit 渲染图表入口：智能推荐图型(可改) + 交互图 + PNG下载"""
    info = suggest_chart(df, question)
    if not info.get("ok"):
        return

    import plotly.express as px

    cat_cols, num_cols = info["cat_cols"], info["num_cols"]

    with st.expander("📊 生成统计图表（系统已推荐图型，可手动切换）", expanded=False):
        c1, c2, c3 = st.columns([1.3, 1, 1])
        chart_type = c1.radio(
            "图型",
            ["line", "bar", "pie"],
            format_func=lambda x: _LABELS[x],
            index=["line", "bar", "pie"].index(info["recommended"]),
            horizontal=True,
            key=f"{key_prefix}_ctype",
        )
        x_col = c2.selectbox("分类 / X轴", cat_cols, key=f"{key_prefix}_x")
        y_col = c3.selectbox("数值 / Y轴", num_cols, key=f"{key_prefix}_y")

        # 自动选第二分类维度做分组着色（如多公司/多口径对比），唯一值2~8才用
        color_col = None
        for c in cat_cols:
            if c != x_col and 2 <= df[c].astype(str).nunique() <= 8:
                color_col = c
                break

        # 清洗数值 + 截断行数
        need_cols = [x_col, y_col] + ([color_col] if color_col else [])
        plot_df = df[need_cols].copy()
        plot_df[y_col] = _to_numeric(plot_df[y_col])
        plot_df = plot_df.dropna(subset=[y_col])
        if plot_df.empty:
            st.caption("⚠️ 所选数值列无法解析为数字，换一列试试")
            return
        if len(plot_df) > _MAX_PLOT_ROWS:
            st.caption(f"数据较多，仅展示前 {_MAX_PLOT_ROWS} 行")
            plot_df = plot_df.head(_MAX_PLOT_ROWS)

        # 饼图不能有负值（亏损等），自动取绝对值并提示
        if chart_type == "pie" and (plot_df[y_col] < 0).any():
            st.caption("⚠️ 含负值，饼图按绝对值展示；如需保留正负请用柱状/折线图")
            plot_df = plot_df.assign(**{y_col: plot_df[y_col].abs()})

        # plotly 交互图（多分类维度时用颜色分组，避免同一X轴堆叠）
        if chart_type == "line":
            fig = px.line(plot_df, x=x_col, y=y_col, color=color_col, markers=True)
        elif chart_type == "pie":
            fig = px.pie(plot_df, names=x_col, values=y_col)
        else:
            fig = px.bar(plot_df, x=x_col, y=y_col, color=color_col,
                         barmode="group", text_auto=".3s")
        # 禁用拖拽缩放，避免图被拖变形；柱状图Y轴从0开始
        fig.update_layout(margin=dict(t=30, b=10, l=10, r=10), height=420,
                          dragmode=False)
        if chart_type in ("line", "bar"):
            fig.update_xaxes(fixedrange=True)
            fig.update_yaxes(fixedrange=True,
                             rangemode="tozero" if chart_type == "bar" else "normal")
        st.plotly_chart(
            fig, use_container_width=True, key=f"{key_prefix}_plot",
            config={"scrollZoom": False, "displayModeBar": False},
        )

        # PNG 下载（matplotlib，中文字体）
        try:
            png = _matplotlib_png(plot_df, chart_type, x_col, y_col)
            st.download_button(
                "⬇️ 下载图表（PNG）",
                png,
                file_name=f"chart_{y_col}.png",
                mime="image/png",
                key=f"{key_prefix}_pngdl",
            )
        except Exception as e:
            st.caption(f"图片下载暂不可用: {e}")


def _matplotlib_png(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str) -> bytes:
    # 同一X有多行时按X聚合求和，避免静态图堆叠误导
    if df[x_col].duplicated().any():
        df = df.groupby(x_col, as_index=False)[y_col].sum()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)
    x = df[x_col].astype(str).tolist()
    y = df[y_col].tolist()

    if chart_type == "line":
        ax.plot(x, y, marker="o", color="#2563eb")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=30, ha="right")
    elif chart_type == "pie":
        ax.pie(y, labels=x, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
    else:
        ax.bar(x, y, color="#3b82f6")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, axis="y", alpha=0.3)
        plt.xticks(rotation=30, ha="right")

    ax.set_title(str(y_col))
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

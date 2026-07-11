"""评估脚本：黄金集驱动的召回评估 + 端到端答案评估

用法：
  python3 evaluate.py --text-retrieval   # 文本召回评估（本地免API）：纯向量 vs 混合召回对比
  python3 evaluate.py --sweep            # 距离门槛标定（本地免API），给出推荐 VEC_DISTANCE_GATE
  python3 evaluate.py --full             # 端到端评估（调DeepSeek API + LLM评审打分）
  python3 evaluate.py --full --limit 5   # 只跑前5题（冒烟测试）

黄金集：eval/golden_set.jsonl（type: text=制度问答 / data=数据问答 / negative=应拒答）
报告：eval/reports/eval_YYYYMMDD_HHMMSS.{json,md}，自动与上一次报告对比给出增减
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

from config import BASE_DIR, VEC_DISTANCE_GATE, VEC_RELATIVE_WINDOW

GOLDEN_PATH = BASE_DIR / "eval" / "golden_set.jsonl"
REPORT_DIR = BASE_DIR / "eval" / "reports"


def load_golden(limit: int = 0) -> list[dict]:
    items = []
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items[:limit] if limit else items


# ============================================================
# 文本召回评估（本地，免API）
# ============================================================
def _apply_gate(hits: list[dict], gate: float, window: float = VEC_RELATIVE_WINDOW) -> list[dict]:
    """复刻 rag_chain 的距离过滤逻辑（重排分数优先，其次距离门槛+相对窗口）"""
    if hits and hits[0].get("rerank_score") is not None:
        from config import RERANK_MIN_SCORE
        return [h for h in hits if h["rerank_score"] >= RERANK_MIN_SCORE]
    filtered = [h for h in hits if h["distance"] < gate]
    if not filtered:
        return []
    best = min(h["distance"] for h in filtered)
    return [h for h in filtered if h["distance"] <= best + window]


def eval_text_retrieval(limit: int = 0):
    from vector_store import VectorStore
    vs = VectorStore()

    items = [g for g in load_golden(limit) if g["type"] in ("text", "negative")]
    texts = [g for g in items if g["type"] == "text"]
    negs = [g for g in items if g["type"] == "negative"]

    methods = {
        "纯向量": lambda q: vs.search(q, top_k=5),
        "混合召回": lambda q: vs.search_hybrid(q, top_k=5),
    }
    results = {}
    for name, fn in methods.items():
        hit, rr_sum, gate_pass = 0, 0.0, 0
        for g in texts:
            hits = fn(g["question"])
            files = [h["metadata"]["source_file"] for h in hits]
            expected = set(g["expected_files"])
            ranks = [i + 1 for i, f in enumerate(files) if f in expected]
            if ranks:
                hit += 1
                rr_sum += 1.0 / ranks[0]
            kept = _apply_gate(hits, VEC_DISTANCE_GATE)
            if any(h["metadata"]["source_file"] in expected for h in kept):
                gate_pass += 1
        neg_reject = sum(
            1 for g in negs if not _apply_gate(fn(g["question"]), VEC_DISTANCE_GATE)
        )
        results[name] = {
            "Hit@5": round(hit / len(texts), 3),
            "MRR": round(rr_sum / len(texts), 3),
            "过门槛率": round(gate_pass / len(texts), 3),
            "负例正确拒答": f"{neg_reject}/{len(negs)}",
        }

    print(f"\n📊 文本召回评估（{len(texts)}正例 + {len(negs)}负例，门槛={VEC_DISTANCE_GATE}）")
    print(f"{'方法':<10}{'Hit@5':<10}{'MRR':<10}{'过门槛率':<12}{'负例拒答':<10}")
    for name, m in results.items():
        print(f"{name:<10}{m['Hit@5']:<10}{m['MRR']:<10}{m['过门槛率']:<12}{m['负例正确拒答']:<10}")
    _save_report("text_retrieval", {"gate": VEC_DISTANCE_GATE, "methods": results,
                                     "n_text": len(texts), "n_negative": len(negs)})
    return results


def eval_sweep(limit: int = 0):
    """距离门槛标定：正例通过率 vs 负例拒答率，推荐两者之和最大的门槛"""
    from vector_store import VectorStore
    vs = VectorStore()

    items = [g for g in load_golden(limit) if g["type"] in ("text", "negative")]
    texts = [g for g in items if g["type"] == "text"]
    negs = [g for g in items if g["type"] == "negative"]

    # 每题只检索一次，扫门槛时复用
    text_hits = [(g, vs.search_hybrid(g["question"], top_k=5)) for g in texts]
    neg_hits = [(g, vs.search_hybrid(g["question"], top_k=5)) for g in negs]

    gates = [round(0.40 + i * 0.025, 3) for i in range(15)]  # 0.40 ~ 0.75
    rows, best = [], (None, -1.0)
    for gate in gates:
        pos_n = sum(
            1 for g, hits in text_hits
            if any(h["metadata"]["source_file"] in set(g["expected_files"])
                   for h in _apply_gate(hits, gate))
        )
        neg_n = sum(1 for _, hits in neg_hits if not _apply_gate(hits, gate))
        pos = pos_n / max(len(texts), 1)
        neg = neg_n / max(len(negs), 1)
        # 按题数加权（答对总题数），避免小样本负例集在比率上被放大权重
        score = pos_n + neg_n
        rows.append({"gate": gate, "正例通过率": round(pos, 3), "负例拒答率": round(neg, 3),
                     "答对题数": f"{pos_n + neg_n}/{len(texts) + len(negs)}"})
        # 同分取更严（更小）的门槛，减少误召回
        if score > best[1]:
            best = (gate, score)

    print(f"\n📊 门槛标定（{len(texts)}正例 + {len(negs)}负例，混合召回，按答对题数推荐）")
    print(f"{'门槛':<8}{'正例通过':<10}{'负例拒答':<10}{'答对题数':<10}")
    for r in rows:
        mark = "  ← 推荐" if r["gate"] == best[0] else ""
        print(f"{r['gate']:<8}{r['正例通过率']:<10}{r['负例拒答率']:<10}{r['答对题数']:<10}{mark}")
    print(f"\n推荐 VEC_DISTANCE_GATE = {best[0]}（当前 config = {VEC_DISTANCE_GATE}）")
    _save_report("sweep", {"rows": rows, "recommended_gate": best[0],
                           "current_gate": VEC_DISTANCE_GATE})
    return best[0]


# ============================================================
# 端到端评估（调API）
# ============================================================
JUDGE_PROMPT = """你是财务问答系统的评审员。根据参考答案评估系统回答的质量，从两个维度打1-5分：
- 准确性：关键事实/数字是否与参考答案一致，有无编造
- 完整性：是否回答了问题的核心

评分标准：5=完全正确且完整；4=正确但略有缺失；3=部分正确；2=大部分错误或答非所问；1=完全错误或编造。
只返回JSON: {"score": 1-5, "reason": "一句话理由"}"""


def _judge(client, model, question, reference, answer) -> dict:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content":
                    f"问题：{question}\n\n参考答案：{reference}\n\n系统回答：{answer[:2000]}"},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        r = json.loads(resp.choices[0].message.content)
        return {"score": int(r.get("score", 0)), "reason": r.get("reason", "")}
    except Exception as e:
        return {"score": 0, "reason": f"评审失败: {e}"}


def _norm(text: str) -> str:
    """数字比对归一化：去千分位逗号和空格"""
    return re.sub(r"[,\s，]", "", text or "")


def eval_full(limit: int = 0):
    from openai import OpenAI
    from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
    from orchestrator import Orchestrator

    orch = Orchestrator()
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    items = load_golden(limit)

    details = []
    for i, g in enumerate(items, 1):
        t0 = time.time()
        try:
            result = orch.ask(g["question"], chat_history=[], stream=False)
            answer = result.get("answer") or ""
            sources = [s.get("file", "") for s in result.get("sources", [])]
        except Exception as e:
            answer, sources, result = f"[执行异常] {e}", [], {}
        latency = round(time.time() - t0, 1)

        d = {"id": g["id"], "type": g["type"], "question": g["question"],
             "latency_s": latency, "answer_head": answer[:150]}

        if g["type"] == "negative":
            markers = ["未找到", "没有找到", "未能找到", "无法找到", "没有相关", "无相关", "未包含"]
            has_marker = any(m in answer[:150] for m in markers)
            # 拒答不得先给无依据的判断（如"不可以。…未找到相关内容"）
            bogus_verdict = answer.strip().startswith(("可以", "不可以", "是的", "不能", "不行"))
            d["正确拒答"] = has_marker and not bogus_verdict
        else:
            expected = set(g.get("expected_files", []))
            d["来源命中"] = expected.issubset(set(sources)) if expected else None
            if g["type"] == "data":
                vals = g.get("expected_values", [])
                d["数值命中"] = all(v in _norm(answer) for v in vals) if vals else None
                d["SQL成功"] = not answer.startswith(("数据查询失败", "[执行异常]"))
            else:
                kws = g.get("expected_keywords", [])
                d["关键词命中"] = all(k.lower() in answer.lower() for k in kws) if kws else None
            j = _judge(client, DEEPSEEK_MODEL, g["question"],
                       g.get("reference_answer", ""), answer)
            d["评审分"] = j["score"]
            d["评审理由"] = j["reason"]

        details.append(d)
        status = d.get("正确拒答", d.get("评审分", "?"))
        print(f"[{i}/{len(items)}] {g['id']} {g['question'][:24]}... → {status} ({latency}s)")

    summary = _summarize_full(details)
    _print_full_summary(summary)
    _save_report("full", {"summary": summary, "details": details}, markdown=True)
    return summary


def _rate(vals: list) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    return f"{sum(vals)}/{len(vals)} ({sum(vals)/len(vals):.0%})"


def _summarize_full(details: list[dict]) -> dict:
    data = [d for d in details if d["type"] == "data"]
    text = [d for d in details if d["type"] == "text"]
    neg = [d for d in details if d["type"] == "negative"]
    scored = [d["评审分"] for d in data + text if d.get("评审分", 0) > 0]
    return {
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "题数": len(details),
        "data": {
            "n": len(data),
            "选表命中": _rate([d.get("来源命中") for d in data]),
            "数值命中": _rate([d.get("数值命中") for d in data]),
            "SQL成功": _rate([d.get("SQL成功") for d in data]),
            "平均评审分": round(sum(d.get("评审分", 0) for d in data) / len(data), 2) if data else 0,
        },
        "text": {
            "n": len(text),
            "引用命中": _rate([d.get("来源命中") for d in text]),
            "关键词命中": _rate([d.get("关键词命中") for d in text]),
            "平均评审分": round(sum(d.get("评审分", 0) for d in text) / len(text), 2) if text else 0,
        },
        "negative": {"n": len(neg), "正确拒答": _rate([d.get("正确拒答") for d in neg])},
        "总平均评审分": round(sum(scored) / len(scored), 2) if scored else 0,
        "平均耗时s": round(sum(d["latency_s"] for d in details) / len(details), 1) if details else 0,
    }


def _print_full_summary(s: dict):
    print("\n" + "=" * 52)
    print(f"📊 端到端评估汇总（{s['题数']}题，{s['时间']}）")
    print(f"数据类({s['data']['n']}): 选表命中 {s['data']['选表命中']} | "
          f"数值命中 {s['data']['数值命中']} | SQL成功 {s['data']['SQL成功']} | "
          f"评审 {s['data']['平均评审分']}/5")
    print(f"制度类({s['text']['n']}): 引用命中 {s['text']['引用命中']} | "
          f"关键词命中 {s['text']['关键词命中']} | 评审 {s['text']['平均评审分']}/5")
    print(f"负例({s['negative']['n']}): 正确拒答 {s['negative']['正确拒答']}")
    print(f"总平均评审分 {s['总平均评审分']}/5 | 平均耗时 {s['平均耗时s']}s")
    prev = _load_prev_summary()
    if prev:
        print(f"（上次 {prev.get('时间','')}: 总平均评审分 {prev.get('总平均评审分','-')}/5）")
    print("=" * 52)


def _load_prev_summary() -> dict | None:
    reports = sorted(REPORT_DIR.glob("eval_full_*.json"))
    if not reports:
        return None
    try:
        return json.loads(reports[-1].read_text(encoding="utf-8")).get("summary")
    except Exception:
        return None


def _save_report(kind: str, payload: dict, markdown: bool = False):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = REPORT_DIR / f"eval_{kind}_{stamp}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                 encoding="utf-8")
    print(f"\n报告已保存: {p.relative_to(BASE_DIR)}")
    if markdown and "details" in payload:
        md = [f"# 端到端评估报告 {stamp}", "", "## 汇总", "```json",
              json.dumps(payload["summary"], ensure_ascii=False, indent=2), "```",
              "", "## 明细", "",
              "| ID | 类型 | 问题 | 结果 | 评审分 | 耗时s |", "|---|---|---|---|---|---|"]
        for d in payload["details"]:
            res = d.get("正确拒答", d.get("数值命中", d.get("关键词命中", "")))
            md.append(f"| {d['id']} | {d['type']} | {d['question'][:20]} "
                      f"| {res} | {d.get('评审分','-')} | {d['latency_s']} |")
        (REPORT_DIR / f"eval_{kind}_{stamp}.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="财务RAG评估")
    ap.add_argument("--text-retrieval", action="store_true", help="文本召回评估（本地免API）")
    ap.add_argument("--sweep", action="store_true", help="距离门槛标定（本地免API）")
    ap.add_argument("--full", action="store_true", help="端到端评估（调API+LLM评审）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前N题")
    args = ap.parse_args()

    if args.sweep:
        eval_sweep(args.limit)
    elif args.full:
        eval_full(args.limit)
    else:
        eval_text_retrieval(args.limit)

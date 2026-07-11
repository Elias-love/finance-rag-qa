"""文本RAG：强制引用出处 + 原文片段 + 反编造"""

from openai import OpenAI
from loguru import logger

from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, TOP_K,
    VEC_DISTANCE_GATE, VEC_RELATIVE_WINDOW, RERANK_MIN_SCORE,
)
from vector_store import VectorStore


RAG_PROMPT = """你是一个严谨的财务知识问答助手。你**只能**根据下方提供的【参考资料】回答问题。

## 强制规则

0. **结论先行（仅限有依据时）**：参考资料能支撑结论时，回答的第一句必须直接给出结论——
   判断类问题先答"可以/不可以/是/否"，数值类问题先给数字，然后再写依据和来源。
   例：「不可以。按考勤制度规定下班时间为18点【来源：…】」，
   而不是「根据考勤制度…因此17点不能下班」。
   **例外（优先级最高）**：如果参考资料不足以回答（触发规则4），第一句就是"未找到"的说明，
   **绝对禁止在前面加"可以/不可以/是/否"等任何判断**——没有依据就没有结论。
1. **只用参考资料回答**：每一句回答都必须有对应的参考资料支撑。绝对不能编造、推测、补充参考资料中没有的内容。
2. **逐条标注来源**：回答中的每个要点后面必须标注出处，格式：
   `【来源：文件名, 第X页, 章节名】`
3. **附原文摘录**：在回答末尾添加"📄 原文摘录"区域，列出你引用的关键原文片段（每段不超过100字），格式：
   > 📄 原文摘录 1（文件名, 第X页）：原文内容...
   > 📄 原文摘录 2（文件名, 第X页）：原文内容...
4. **参考资料不足时**：明确回答"根据当前知识库中的资料，未找到与该问题直接相关的内容。建议上传相关文档后再次查询。"，不要尝试用常识补充。
5. **使用简体中文回答**。
6. **回答要准确、专业、简洁**，不要加无关内容。
7. **关于截图/系统界面图**：参考资料是从文档中提取的**文字**，本身不含图片，但若某条参考资料标注了"【本文件含N张系统截图】"，说明系统会自动在你回答的下方附上这些截图。此时：
   - 绝对不要说"参考资料不含截图""未找到截图"之类的话；
   - 应正常回答文字步骤，并在结尾提示"相关系统截图见下方📷"。"""


class RAGChain:
    def __init__(self):
        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self.vector_store = VectorStore()
        from image_store import ImageStore
        self.image_store = ImageStore()

    def _file_image_counts(self, hits: list[dict]) -> dict:
        """统计 hits 涉及的各文件含多少张截图"""
        import sqlite3
        from config import DB_PATH
        files = {h["metadata"]["source_file"] for h in hits}
        counts = {}
        with sqlite3.connect(DB_PATH) as conn:
            for f in files:
                n = conn.execute(
                    "SELECT COUNT(*) FROM page_images WHERE source_file = ?", (f,)
                ).fetchone()[0]
                if n:
                    counts[f] = n
        return counts

    def query(self, question: str, top_k: int = TOP_K) -> dict:
        # 混合召回：向量+BM25双路 → RRF融合 → （模型可用时）交叉编码器重排
        hits = self.vector_store.search_hybrid(question, top_k=top_k)

        if not hits:
            return {
                "answer": "根据当前知识库中的资料，未找到与该问题直接相关的内容。请确认是否已上传相关文档。",
                "sources": [],
                "snippets": [],
            }

        filtered = self._filter_hits(hits)
        if not filtered:
            return {
                "answer": "根据当前知识库中的资料，未找到与该问题直接相关的内容。建议上传相关文档后再次查询。",
                "sources": [],
                "snippets": [],
            }

        img_counts = self._file_image_counts(filtered)
        context = self._build_context(filtered, img_counts)

        response = self.client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": RAG_PROMPT},
                {"role": "user", "content": f"【参考资料】\n{context}\n\n【问题】{question}"},
            ],
            temperature=0,
        )

        answer = response.choices[0].message.content

        # 答案与来源保持一致：LLM 判定未找到时，不展示任何引用
        if self._is_not_found(answer):
            logger.info(f"RAG回答: {question[:30]}... → LLM判定未找到，清空引用")
            return {"answer": answer, "sources": [], "snippets": []}

        sources = self._extract_sources(filtered)
        snippets = self._extract_snippets(filtered)

        logger.info(f"RAG回答: {question[:30]}... → 引用{len(sources)}个来源, {len(snippets)}段原文")
        return {"answer": answer, "sources": sources, "snippets": snippets}

    @staticmethod
    def _filter_hits(hits: list[dict]) -> list[dict]:
        """相关性过滤。阈值在 config.py，由 evaluate.py --sweep 在黄金集上标定。

        - 重排器可用（有 rerank_score）：按重排分数过滤，这是最准的信号
        - 否则：向量距离绝对门槛 + 相对窗口（丢弃与最佳命中差距过大的）
        """
        if hits and hits[0].get("rerank_score") is not None:
            return [h for h in hits if h["rerank_score"] >= RERANK_MIN_SCORE]

        filtered = [h for h in hits if h["distance"] < VEC_DISTANCE_GATE]
        if not filtered:
            return []
        best = min(h["distance"] for h in filtered)
        return [h for h in filtered if h["distance"] <= best + VEC_RELATIVE_WINDOW]

    @staticmethod
    def _is_not_found(answer: str) -> bool:
        """检测 LLM 回答是否表示'未找到相关内容'"""
        if not answer:
            return True
        # 回答中带引用标注（提示词强制格式【来源：…】）说明给出了有依据的内容，
        # 不算"未找到"——避免"制度中未找到直接规定，但第X条…【来源：…】"
        # 这类有实质内容的回答被整体清空引用
        if "【来源" in answer:
            return False
        markers = ["未找到", "没有找到", "未能找到", "无法找到", "没有与该问题",
                   "未包含", "没有相关", "无相关"]
        head = answer[:120]
        return any(m in head for m in markers)

    def _build_context(self, hits: list[dict], img_counts: dict = None) -> str:
        img_counts = img_counts or {}
        parts = []
        for i, hit in enumerate(hits, 1):
            meta = hit["metadata"]
            fname = meta["source_file"]
            img_hint = f"【本文件含{img_counts[fname]}张系统截图】" if fname in img_counts else ""

            # 构建来源标签
            source_parts = [meta["source_file"]]
            if meta.get("page_num"):
                source_parts.append(f"第{meta['page_num']}页")
            if meta.get("heading_path"):
                source_parts.append(meta["heading_path"])
            elif meta.get("section"):
                source_parts.append(meta["section"])
            source_label = ", ".join(source_parts)

            # 块类型标注
            block_type = meta.get("block_type", "")
            type_hint = ""
            if "footnote" in block_type:
                type_hint = "（脚注）"
            elif "ocr" in block_type:
                type_hint = "（OCR识别）"

            parts.append(
                f"【参考{i}】[{source_label}]{type_hint}{img_hint}\n{hit['content']}"
            )
        return "\n\n---\n\n".join(parts)

    def _extract_sources(self, hits: list[dict]) -> list[dict]:
        seen = set()
        sources = []
        for hit in hits:
            meta = hit["metadata"]
            key = f"{meta['source_file']}:{meta.get('page_num', 0)}:{meta.get('section', '')}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "file": meta["source_file"],
                    "page": meta.get("page_num", 0),
                    "section": meta.get("section", ""),
                    "heading_path": meta.get("heading_path", ""),
                    "block_type": meta.get("block_type", ""),
                    "relevance": round(1 - hit["distance"], 3),
                })
        return sources

    def _extract_snippets(self, hits: list[dict]) -> list[dict]:
        """提取原文片段，供前端展示。展示前做严格清洗，过滤所有可疑token。"""
        from document_processor import PDFStructureParser
        parser = PDFStructureParser()

        snippets = []
        for hit in hits:
            meta = hit["metadata"]
            content = hit["content"]
            # 严格清洗：丢弃所有可疑token，只保留确认正常的文本
            cleaned = self._strict_clean(content, parser)

            if not cleaned or len(cleaned.strip()) < 10:
                # 全段乱码，跳过该片段
                continue

            snippet_text = cleaned[:300] + ("…" if len(cleaned) > 300 else "")
            snippets.append({
                "text": snippet_text,
                "file": meta["source_file"],
                "page": meta.get("page_num", 0),
                "section": meta.get("section", ""),
                "heading_path": meta.get("heading_path", ""),
                "relevance": round(1 - hit["distance"], 3),
            })
        return snippets

    # 业务术语白名单（不会被当作乱码丢弃）
    _BIZ_WHITELIST = {
        "SAP", "ERP", "OA", "HR", "CEO", "CFO", "CTO", "VP", "PM", "PMO",
        "FY", "Q1", "Q2", "Q3", "Q4", "AR", "AP", "GL", "PL", "BS", "CF",
        "USD", "CNY", "EUR", "HKD", "RMB", "JPY", "ID", "NO",
        "TB", "PR", "PO", "WMS", "CRM", "MES", "BOM", "BI",
    }

    @classmethod
    def _is_normal_token(cls, tok: str) -> bool:
        """白名单：明确正常的 token 才返回 True"""
        import re as _re
        if not tok:
            return False

        # 纯标点
        if len(tok) <= 3 and all(c in "，。；：、！？,.;:!?\"'（）()【】[]《》<>—-…" for c in tok):
            return True

        # 业务术语
        if tok.upper() in cls._BIZ_WHITELIST:
            return True

        # 纯数字 (>=1位)
        if tok.isdigit():
            return True

        # 日期/金额格式: 2025-01-01, 1,234.56, 30%, 12:00
        if _re.fullmatch(r"[\d,./:%\-+]+", tok):
            return True

        # 标准事务码：小写字母开头+字母数字（如 a3p9130, aapt430）
        if _re.fullmatch(r"[a-z][a-z\d]{3,12}", tok):
            return True

        # 纯中文 ≥2 字
        cjk_count = sum(1 for c in tok if "一" <= c <= "鿿")
        if cjk_count >= 2 and cjk_count == len(tok):
            return True

        # 单中文字（如"的"、"是"，配合上下文可能有意义）
        if cjk_count == 1 and len(tok) == 1:
            return True

        # 中文+数字/字母短缀（如"1.1应付账款"、"OA系统"、"SAP环境"）
        # 模式: 数字/字母前缀 + 中文连续 ≥ 2字
        if _re.fullmatch(r"[\d.A-Za-z]{1,6}[一-鿿]{2,}", tok):
            return True
        # 模式: 中文连续 ≥ 2字 + 数字/字母后缀
        if _re.fullmatch(r"[一-鿿]{2,}[\d.A-Za-z]{1,6}", tok):
            return True

        # 完整英文单词（小写≥3字符）
        if _re.fullmatch(r"[a-z]{3,}", tok):
            return True

        # 含连字符的产品名/版本（如 V1.0, Q4-2025）
        if _re.fullmatch(r"[A-Z]\d+", tok) or _re.fullmatch(r"[A-Za-z]\d+\.\d+", tok):
            return True

        return False

    def _strict_clean(self, text: str, parser) -> str:
        """按完整句子评估：中文占比足够才保留"""
        import re as _re
        if not text:
            return ""

        # 把整段（含\n）按中英文标点切成句子
        # 切分点：中文句号/感叹/问号/分号 + 换行
        sentences = _re.split(r"(?<=[。！？；\n])", text)

        kept_sents = []
        for s in sentences:
            s_strip = s.strip()
            if not s_strip:
                continue
            if len(s_strip) < 3:
                continue

            # 先按token白名单过滤一遍
            tokens = s_strip.split()
            kept_tokens = [t for t in tokens if self._is_normal_token(t)]
            if not kept_tokens:
                continue

            # 裁掉句首/句尾连续的"非中文 token"碎片（中文是有意义内容的核心）
            def _has_cjk(tok):
                return any("一" <= c <= "鿿" for c in tok)

            # 句首
            while kept_tokens and not _has_cjk(kept_tokens[0]):
                kept_tokens.pop(0)
            # 句尾
            while kept_tokens and not _has_cjk(kept_tokens[-1]):
                kept_tokens.pop()

            if not kept_tokens:
                continue

            # 移除句子中间连续 >=3 个 非中文 token 的孤立段（如 "X. 9141 51.53 22." 这种）
            i = 0
            filtered = []
            while i < len(kept_tokens):
                t = kept_tokens[i]
                if _has_cjk(t):
                    filtered.append(t)
                    i += 1
                else:
                    # 看连续多少个非中文
                    j = i
                    while j < len(kept_tokens) and not _has_cjk(kept_tokens[j]):
                        j += 1
                    non_cjk_run = j - i
                    if non_cjk_run >= 3:
                        # 整段非中文且无后续中文衔接，丢弃
                        pass
                    else:
                        # 短段（1-2个）保留作为数字/缩写连接
                        filtered.extend(kept_tokens[i:j])
                    i = j

            if not filtered:
                continue

            kept_text = " ".join(filtered)
            # 统计字符比例
            n = len(_re.sub(r"\s+", "", kept_text))
            if n < 5:
                continue
            cjk = sum(1 for c in kept_text if "一" <= c <= "鿿")
            # 中文占比 >= 50% 才认为是有效语义内容
            if cjk / n < 0.50:
                continue

            kept_sents.append(kept_text)

        return " ".join(kept_sents).strip()

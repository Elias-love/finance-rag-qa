"""混合召回层：BM25(字符二元组) + 向量召回 → RRF融合 → 可选交叉编码器重排

设计要点：
- BM25 纯 Python 实现，零新增依赖；中文用字符二元组分词，免去 jieba
- 融合用 Reciprocal Rank Fusion（RRF），对两路打分尺度不敏感
- BM25 独有命中会补算真实向量距离，下游的距离门槛逻辑对所有命中统一生效
- 重排模型（bge-reranker）本地有缓存才启用，没有则静默退化为 RRF 排序
"""

import hashlib
import math
import re
from collections import Counter

from loguru import logger

from config import RRF_K, RECALL_K, RERANK_MODEL, RERANK_ENABLED


# ============================================================
# BM25（Okapi），字符二元组分词
# ============================================================
def tokenize(text: str) -> list[str]:
    """中文按字符二元组，英文/数字按小写整词。
    例："应付账款ap对账" → ["应付","付账","账款","ap","对账"]
    """
    if not text:
        return []
    tokens = []
    # 英文单词/数字串整体保留（含 aapt110 这类事务码）
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9_]*|\d[\d.,%]*", text):
        tokens.append(m.group().lower())
    # 中文连续段切二元组
    for m in re.finditer(r"[一-鿿]+", text):
        seg = m.group()
        if len(seg) == 1:
            tokens.append(seg)
        else:
            tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    return tokens


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.ids: list[str] = []
        self.doc_tf: list[Counter] = []
        self.doc_len: list[int] = []
        self.df: Counter = Counter()
        self.avg_len = 0.0
        self.n_docs = 0

    def build(self, ids: list[str], docs: list[str]):
        self.ids = list(ids)
        self.doc_tf, self.doc_len = [], []
        self.df = Counter()
        for doc in docs:
            toks = tokenize(doc)
            tf = Counter(toks)
            self.doc_tf.append(tf)
            self.doc_len.append(len(toks))
            for term in tf:
                self.df[term] += 1
        self.n_docs = len(docs)
        self.avg_len = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """返回 [(doc_id, bm25_score)]，按分数降序，零分不返回"""
        if not self.n_docs:
            return []
        q_terms = set(tokenize(query))
        scores = [0.0] * self.n_docs
        for term in q_terms:
            df = self.df.get(term)
            if not df:
                continue
            idf = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(self.doc_tf):
                f = tf.get(term)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
                scores[i] += idf * f * (self.k1 + 1) / denom
        ranked = [(self.ids[i], s) for i, s in enumerate(scores) if s > 0]
        ranked.sort(key=lambda x: -x[1])
        return ranked[:top_k]


# ============================================================
# RRF 融合
# ============================================================
def rrf_fuse(rank_lists: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """多路排名融合：score(d) = Σ 1/(k+rank)，rank 从 1 开始"""
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for r, doc_id in enumerate(ranks, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + r)
    return sorted(scores.items(), key=lambda x: -x[1])


# ============================================================
# 可选重排器
# ============================================================
class Reranker:
    """bge-reranker 交叉编码器。模型本地无缓存时 available=False，调用方退化处理。"""

    def __init__(self):
        self.model = None
        if str(RERANK_ENABLED) == "0":
            return
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(RERANK_MODEL, max_length=512)
            logger.info(f"重排器已启用: {RERANK_MODEL}")
        except Exception as e:
            if str(RERANK_ENABLED) == "1":
                logger.warning(f"重排器加载失败（RERANK_ENABLED=1 但模型不可用）: {e}")
            else:
                logger.info(f"重排器未启用（模型无本地缓存），退化为RRF排序: {type(e).__name__}")

    @property
    def available(self) -> bool:
        return self.model is not None

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        """返回每个 doc 与 query 的相关性分数（sigmoid 归一化到 0-1）"""
        raw = self.model.predict([(query, d) for d in docs])
        return [1.0 / (1.0 + math.exp(-float(s))) for s in raw]


# ============================================================
# 混合检索器
# ============================================================
class HybridRetriever:
    """向量 + BM25 双路召回，RRF 融合，可选重排。

    对外返回与原 VectorStore.search 相同结构的 hits（content/metadata/distance），
    额外附带 rrf_score / bm25_score / rerank_score / channels 字段。
    """

    def __init__(self, collection, embedding_fn):
        self.collection = collection
        self.embedding_fn = embedding_fn
        self.bm25 = BM25Index()
        self._built_count = -1
        self._built_fingerprint = None  # ids 指纹：覆盖替换后 chunk 数不变也能检测到变化
        self._doc_cache: dict[str, dict] = {}
        self.reranker = Reranker()

    def _ensure_index(self):
        # 失效检测用 ids 指纹而非数量：覆盖替换同结构文件时 chunk 总数往往不变，
        # 只比较 count 会沿用旧索引和旧 _doc_cache，返回已删除文件的内容
        ids_now = self.collection.get(include=[])["ids"]
        fingerprint = hashlib.md5("\n".join(sorted(ids_now)).encode()).hexdigest()
        if fingerprint == self._built_fingerprint:
            return
        data = self.collection.get(include=["documents", "metadatas"])
        ids = data["ids"]
        docs = data["documents"]
        self.bm25.build(ids, docs)
        self._doc_cache = {
            i: {"content": d, "metadata": m}
            for i, d, m in zip(ids, docs, data["metadatas"])
        }
        self._built_count = len(ids)
        self._built_fingerprint = fingerprint
        logger.info(f"BM25索引已构建: {len(ids)}个chunk")

    @staticmethod
    def _cosine_distance(a, b) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 1.0
        return 1.0 - dot / (na * nb)

    def search(self, query: str, top_k: int = 5, recall_k: int = RECALL_K) -> list[dict]:
        self._ensure_index()
        if self._built_count == 0:
            return []

        # ── 双路召回 ──
        vec_res = self.collection.query(query_texts=[query], n_results=min(recall_k, self._built_count))
        vec_ids = vec_res["ids"][0]
        vec_dist = {i: d for i, d in zip(vec_ids, vec_res["distances"][0])}

        bm25_hits = self.bm25.search(query, top_k=recall_k)
        bm25_ids = [i for i, _ in bm25_hits]
        bm25_score = dict(bm25_hits)

        # ── RRF 融合 ──
        fused = rrf_fuse([vec_ids, bm25_ids])
        candidate_ids = [i for i, _ in fused[: max(top_k * 2, top_k)]]
        rrf_score = dict(fused)

        # ── BM25独有命中补算真实向量距离，统一下游门槛逻辑 ──
        missing = [i for i in candidate_ids if i not in vec_dist]
        if missing:
            got = self.collection.get(ids=missing, include=["embeddings"])
            q_emb = self.embedding_fn([query])[0]
            for i, emb in zip(got["ids"], got["embeddings"]):
                vec_dist[i] = self._cosine_distance(q_emb, emb)

        hits = []
        for cid in candidate_ids:
            doc = self._doc_cache.get(cid)
            if not doc:
                continue
            hits.append({
                "content": doc["content"],
                "metadata": doc["metadata"],
                "distance": vec_dist.get(cid, 1.0),
                "rrf_score": rrf_score.get(cid, 0.0),
                "bm25_score": bm25_score.get(cid, 0.0),
                "rerank_score": None,
                "channels": (["vector"] if cid in set(vec_ids) else [])
                            + (["bm25"] if cid in bm25_score else []),
            })

        # ── 可选重排 ──
        if self.reranker.available and hits:
            scores = self.reranker.rerank(query, [h["content"] for h in hits])
            for h, s in zip(hits, scores):
                h["rerank_score"] = s
            hits.sort(key=lambda h: -h["rerank_score"])

        return hits[:top_k]

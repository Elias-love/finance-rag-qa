"""hybrid_retriever.py 单元测试：分词 + BM25 + RRF融合 + rag_chain过滤逻辑
运行：python3 -m unittest discover -s tests
"""
import unittest

from hybrid_retriever import tokenize, BM25Index, rrf_fuse
from rag_chain import RAGChain
from config import VEC_DISTANCE_GATE, RERANK_MIN_SCORE


class TestTokenize(unittest.TestCase):
    def test_chinese_bigrams(self):
        self.assertEqual(tokenize("应付账款"), ["应付", "付账", "账款"])

    def test_english_and_codes_kept_whole(self):
        toks = tokenize("在aapt110审核发票")
        self.assertIn("aapt110", toks)
        self.assertIn("审核", toks)

    def test_mixed_case_lowered(self):
        self.assertIn("capp004", tokenize("Capp004 取消付款"))

    def test_empty(self):
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize(None), [])

    def test_single_chinese_char(self):
        self.assertEqual(tokenize("好"), ["好"])


class TestBM25(unittest.TestCase):
    def setUp(self):
        self.idx = BM25Index()
        self.idx.build(
            ["d1", "d2", "d3"],
            [
                "应付账款对账数据在aapq130导出，账期为上月26号到本月25号",
                "应付月结作业aapp910，反月结用aapp970",
                "财务部考勤制度：上班时间早上8点30到下午18点",
            ],
        )

    def test_exact_code_ranks_first(self):
        # 精确事务码匹配——这正是BM25补足向量检索的场景
        hits = self.idx.search("aapq130在哪里", top_k=3)
        self.assertEqual(hits[0][0], "d1")

    def test_semantic_terms(self):
        hits = self.idx.search("月结作业", top_k=3)
        self.assertEqual(hits[0][0], "d2")

    def test_no_match_returns_empty(self):
        self.assertEqual(self.idx.search("量子物理", top_k=3), [])

    def test_empty_index(self):
        empty = BM25Index()
        self.assertEqual(empty.search("任何问题"), [])


class TestRRF(unittest.TestCase):
    def test_doc_in_both_lists_wins(self):
        fused = rrf_fuse([["a", "b", "c"], ["b", "d"]])
        self.assertEqual(fused[0][0], "b")  # b 在两路都命中

    def test_single_list_preserves_order(self):
        fused = rrf_fuse([["x", "y"]])
        self.assertEqual([d for d, _ in fused], ["x", "y"])

    def test_empty(self):
        self.assertEqual(rrf_fuse([]), [])
        self.assertEqual(rrf_fuse([[], []]), [])


def _hit(distance, rerank=None, f="a.docx"):
    return {"content": "x", "metadata": {"source_file": f},
            "distance": distance, "rerank_score": rerank}


class TestFilterHits(unittest.TestCase):
    def test_distance_gate(self):
        hits = [_hit(0.3), _hit(VEC_DISTANCE_GATE + 0.1)]
        kept = RAGChain._filter_hits(hits)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["distance"], 0.3)

    def test_all_beyond_gate_empty(self):
        self.assertEqual(RAGChain._filter_hits([_hit(0.9), _hit(0.95)]), [])

    def test_relative_window(self):
        # 0.30 和 0.35 在窗口内，0.50 掉队被剔除
        hits = [_hit(0.30), _hit(0.35), _hit(0.50)]
        kept = RAGChain._filter_hits(hits)
        self.assertEqual({h["distance"] for h in kept}, {0.30, 0.35})

    def test_rerank_score_takes_priority(self):
        # 有重排分数时按重排分数过滤，距离不再起作用
        hits = [_hit(0.9, rerank=0.8), _hit(0.2, rerank=RERANK_MIN_SCORE - 0.1)]
        kept = RAGChain._filter_hits(hits)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["rerank_score"], 0.8)

    def test_empty_input(self):
        self.assertEqual(RAGChain._filter_hits([]), [])


if __name__ == "__main__":
    unittest.main()

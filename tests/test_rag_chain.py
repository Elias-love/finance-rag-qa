"""rag_chain.py 单元测试：未找到判定 + token 白名单 + 乱码清洗
运行：python3 -m unittest discover -s tests

注：RAGChain.__init__ 会加载向量库和嵌入模型，测试用 object.__new__ 跳过，
只测不依赖外部资源的纯函数。
"""
import unittest

from rag_chain import RAGChain


def _bare_chain():
    return object.__new__(RAGChain)


class TestIsNotFound(unittest.TestCase):
    def test_not_found_markers(self):
        self.assertTrue(RAGChain._is_not_found("根据当前知识库中的资料，未找到与该问题直接相关的内容。"))
        self.assertTrue(RAGChain._is_not_found("知识库中没有相关内容。"))

    def test_empty_answer_is_not_found(self):
        self.assertTrue(RAGChain._is_not_found(""))
        self.assertTrue(RAGChain._is_not_found(None))

    def test_normal_answer(self):
        self.assertFalse(RAGChain._is_not_found("报销流程分三步：填单、审批、打款。"))

    def test_marker_beyond_head_ignored(self):
        # 标记只在前120字符内生效，正文靠后出现"未找到"不误判
        answer = "报销流程如下。" + "详" * 120 + "若未找到单据请联系财务。"
        self.assertFalse(RAGChain._is_not_found(answer))


class TestIsNormalToken(unittest.TestCase):
    def test_business_terms(self):
        for tok in ["SAP", "ERP", "OA", "USD", "TB"]:
            self.assertTrue(RAGChain._is_normal_token(tok), tok)

    def test_pure_chinese(self):
        self.assertTrue(RAGChain._is_normal_token("净利润"))
        self.assertTrue(RAGChain._is_normal_token("的"))

    def test_numbers_dates_amounts(self):
        for tok in ["2025", "2025-01-01", "1,234.56", "30%"]:
            self.assertTrue(RAGChain._is_normal_token(tok), tok)

    def test_mixed_chinese_affix(self):
        self.assertTrue(RAGChain._is_normal_token("1.1应付账款"))
        self.assertTrue(RAGChain._is_normal_token("OA系统"))

    def test_garbled_rejected(self):
        for tok in ["▲◆●", "ÿþ", "", "§¶†"]:
            self.assertFalse(RAGChain._is_normal_token(tok), repr(tok))


class TestStrictClean(unittest.TestCase):
    def setUp(self):
        self.chain = _bare_chain()

    def test_chinese_content_kept(self):
        text = "报销流程 需要部门经理审批 然后提交财务"
        out = self.chain._strict_clean(text, None)
        self.assertIn("报销流程", out)
        self.assertIn("部门经理审批", out)

    def test_garbled_only_dropped(self):
        text = "▲◆● ÿþ§ 9141 51.53 22. ¶†‡"
        out = self.chain._strict_clean(text, None)
        self.assertEqual(out, "")

    def test_empty_input(self):
        self.assertEqual(self.chain._strict_clean("", None), "")
        self.assertEqual(self.chain._strict_clean(None, None), "")

    def test_mixed_keeps_chinese_drops_garbled_run(self):
        # 中文句子保留；句中连续>=3个非中文碎片段被剔除
        text = "应付账款 对账流程 ▲◆ 9141 51.53 22.8 ¶† 每月执行 一次核对"
        out = self.chain._strict_clean(text, None)
        self.assertIn("对账流程", out)
        self.assertIn("每月执行", out)
        self.assertNotIn("9141", out)


if __name__ == "__main__":
    unittest.main()

"""orchestrator.py 单元测试：追问判定 + 对话历史预算
运行：python3 -m unittest discover -s tests

注：Orchestrator.__init__ 会初始化向量库/嵌入模型，测试用 object.__new__ 跳过。
"""
import unittest

from orchestrator import Orchestrator


def _bare():
    return object.__new__(Orchestrator)


class TestIsFollowup(unittest.TestCase):
    def setUp(self):
        self.orch = _bare()

    def test_pronoun_is_followup(self):
        self.assertTrue(self.orch._is_followup("这个数字是怎么算出来的？"))

    def test_short_question_is_followup(self):
        self.assertTrue(self.orch._is_followup("净利润呢"))

    def test_long_standalone_not_followup(self):
        q = "2025年深圳星辰数字科技集团股份有限公司的营业收入是多少"
        self.assertFalse(self.orch._is_followup(q))


class TestHistoryBudget(unittest.TestCase):
    def setUp(self):
        self.orch = _bare()

    def test_budget_fits_model_context(self):
        # DeepSeek 上下文 64K tokens，中文约 1 token≈1.5字符。
        # 历史预算折算成 tokens 后必须显著低于总上下文（留给系统提示词/表结构/结果）
        est_tokens = Orchestrator.MAX_HISTORY_CHARS / 1.5
        self.assertLess(est_tokens, 64_000 * 0.6,
                        "历史预算过大会撑爆模型上下文（此前 120_000 字符≈8万tokens 即超限）")

    def test_empty_history(self):
        self.assertEqual(self.orch._build_history_context([]), "")
        self.assertEqual(self.orch._build_history_context(None), "")

    def test_truncates_oldest_first(self):
        big = Orchestrator.MAX_HISTORY_CHARS // 2
        history = [
            {"role": "user", "content": "A" * big},        # 最旧，应被丢弃
            {"role": "assistant", "content": "B" * big},
            {"role": "user", "content": "C" * 100},        # 最新，必须保留
        ]
        ctx = self.orch._build_history_context(history)
        self.assertLessEqual(len(ctx), Orchestrator.MAX_HISTORY_CHARS + 200)
        self.assertIn("C" * 100, ctx)
        self.assertNotIn("A" * 10, ctx)

    def test_chronological_order_preserved(self):
        history = [
            {"role": "user", "content": "第一句"},
            {"role": "assistant", "content": "第二句"},
        ]
        ctx = self.orch._build_history_context(history)
        self.assertLess(ctx.index("第一句"), ctx.index("第二句"))

    def test_none_content_tolerated(self):
        history = [{"role": "assistant", "content": None},
                   {"role": "user", "content": "问题"}]
        ctx = self.orch._build_history_context(history)
        self.assertIn("问题", ctx)


class TestDfPreview(unittest.TestCase):
    """回归：同列多量级数值时 pandas 默认切科学计数法，LLM会拿到残缺数字"""

    def test_mixed_magnitude_no_scientific_notation(self):
        import pandas as pd
        df = pd.DataFrame({
            "项目": ["单体营业收入", "合并营业收入"],
            "本期数": [12345678.99, 9876543210.12],
        })
        out = Orchestrator._df_preview(df)
        self.assertNotIn("e+", out)
        self.assertIn("12,345,678.99", out)
        self.assertIn("9,876,543,210.12", out)

    def test_negative_and_zero(self):
        import pandas as pd
        df = pd.DataFrame({"项目": ["净利润", "收入"], "本期数": [-4321.09, 0.0]})
        out = Orchestrator._df_preview(df)
        self.assertIn("-4,321.09", out)
        self.assertIn("0.00", out)


if __name__ == "__main__":
    unittest.main()

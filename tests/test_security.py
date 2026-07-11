"""security.py 单元测试：敏感信息脱敏 + 问题安全检查
运行：python3 -m unittest discover -s tests
"""
import unittest

import pandas as pd

from security import SecurityManager


class TestMaskSensitive(unittest.TestCase):
    def setUp(self):
        self.sec = SecurityManager()

    def test_mask_id_card(self):
        out = self.sec.mask_sensitive("员工身份证110101199003074567已登记")
        self.assertIn("****身份证****", out)
        self.assertNotIn("110101199003074567", out)

    def test_mask_id_card_with_x(self):
        out = self.sec.mask_sensitive("证件号44030119851201123X")
        self.assertIn("****身份证****", out)

    def test_mask_bank_card_with_context(self):
        out = self.sec.mask_sensitive("卡号：6222020200066888888")
        self.assertIn("****银行卡****", out)
        self.assertNotIn("6222020200066888888", out)

    def test_mask_standalone_19_digit(self):
        out = self.sec.mask_sensitive("收款账户 6222021234567890123 请核对")
        self.assertIn("****银行卡****", out)

    def test_16_digit_order_number_not_masked(self):
        # 独立16位数字是订单号/流水号，不应误伤
        out = self.sec.mask_sensitive("订单 1234567890123456 已发货")
        self.assertIn("1234567890123456", out)

    def test_mask_phone(self):
        out = self.sec.mask_sensitive("联系电话13812345678")
        self.assertIn("****手机****", out)
        self.assertNotIn("13812345678", out)

    def test_amount_not_masked(self):
        # 普通金额不应被脱敏
        out = self.sec.mask_sensitive("营业收入 269,000,000.00 元")
        self.assertIn("269,000,000.00", out)

    def test_non_string_passthrough(self):
        self.assertEqual(self.sec.mask_sensitive(12345), 12345)


class TestMaskDataframe(unittest.TestCase):
    def setUp(self):
        self.sec = SecurityManager()

    def test_text_column_masked_numeric_kept(self):
        df = pd.DataFrame({
            "备注": ["电话13812345678", "正常备注"],
            "金额": [1000.5, 2000.0],
        })
        out = self.sec.mask_dataframe(df)
        self.assertIn("****手机****", out["备注"][0])
        self.assertEqual(out["金额"][0], 1000.5)
        # 原 df 不被修改
        self.assertIn("13812345678", df["备注"][0])

    def test_empty_df(self):
        df = pd.DataFrame()
        self.assertTrue(self.sec.mask_dataframe(df).empty)

    def test_none_passthrough(self):
        self.assertIsNone(self.sec.mask_dataframe(None))


class TestQuestionSafety(unittest.TestCase):
    def setUp(self):
        self.sec = SecurityManager()

    def test_sensitive_field_warned(self):
        r = self.sec.check_question_safety("查一下员工的身份证号")
        self.assertFalse(r["safe"])
        self.assertIn("身份证", r["warning"])

    def test_normal_question_safe(self):
        r = self.sec.check_question_safety("2025年营业收入是多少")
        self.assertTrue(r["safe"])


if __name__ == "__main__":
    unittest.main()

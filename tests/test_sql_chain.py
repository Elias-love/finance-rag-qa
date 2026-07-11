"""sql_chain.py 单元测试：SQL校验/自动修复 + 口径/币种补全 + 标签生成
运行：python3 -m unittest discover -s tests
"""
import unittest

from sql_chain import SQLChain


def _make_chain():
    # __init__ 仅创建 OpenAI 客户端对象，不发起网络请求
    return SQLChain()


class TestValidateSQL(unittest.TestCase):
    def setUp(self):
        self.chain = _make_chain()

    def test_select_allowed(self):
        r = self.chain._validate_sql('SELECT "项目","本期数" FROM "t_利润表"')
        self.assertTrue(r["safe"])

    def test_empty_rejected(self):
        self.assertFalse(self.chain._validate_sql("")["safe"])
        self.assertFalse(self.chain._validate_sql("   ")["safe"])

    def test_drop_rejected(self):
        self.assertFalse(self.chain._validate_sql("DROP TABLE t")["safe"])

    def test_update_rejected(self):
        self.assertFalse(self.chain._validate_sql("UPDATE t SET a=1")["safe"])

    def test_multi_statement_rejected(self):
        r = self.chain._validate_sql("SELECT * FROM t; DELETE FROM t")
        self.assertFalse(r["safe"])

    def test_trailing_semicolon_allowed(self):
        r = self.chain._validate_sql("SELECT * FROM t;")
        self.assertTrue(r["safe"])

    def test_comment_rejected(self):
        self.assertFalse(self.chain._validate_sql("SELECT * FROM t -- x")["safe"])
        self.assertFalse(self.chain._validate_sql("SELECT /* x */ * FROM t")["safe"])

    def test_chinese_column_with_blocked_substring_ok(self):
        # 中文列名不会误触发英文关键词黑名单
        r = self.chain._validate_sql('SELECT "本期数" FROM "t_资产负债表"')
        self.assertTrue(r["safe"])


class TestFixTypos(unittest.TestCase):
    def setUp(self):
        self.chain = _make_chain()

    def test_common_typos_fixed(self):
        fixed = self.chain._fix_typos("SELCT * FORM t WHRE a=1")
        self.assertIn("SELECT", fixed)
        self.assertIn("FROM", fixed)
        self.assertIn("WHERE", fixed)

    def test_correct_sql_untouched(self):
        sql = 'SELECT "项目" FROM "t" WHERE "项目" LIKE \'%净利润%\''
        self.assertEqual(self.chain._fix_typos(sql), sql)

    def test_word_boundary_respected(self):
        # FORM 只在独立单词时替换，不改 "PLATFORM" 这类子串
        fixed = self.chain._fix_typos("SELECT * FROM PLATFORM_T")
        self.assertIn("PLATFORM_T", fixed)


class TestFixUnionLimit(unittest.TestCase):
    def setUp(self):
        self.chain = _make_chain()

    def test_branch_limits_moved(self):
        sql = ("SELECT a FROM t1 LIMIT 100 UNION ALL SELECT a FROM t2 LIMIT 100")
        fixed = self.chain._fix_union_limit(sql)
        # 只保留最后一个 LIMIT
        self.assertEqual(fixed.upper().count("LIMIT"), 1)
        self.assertTrue(fixed.rstrip().upper().endswith("LIMIT 100"))

    def test_no_union_untouched(self):
        sql = "SELECT a FROM t1 LIMIT 100"
        self.assertEqual(self.chain._fix_union_limit(sql), sql)

    def test_single_limit_untouched(self):
        sql = "SELECT a FROM t1 UNION ALL SELECT a FROM t2 LIMIT 100"
        self.assertEqual(self.chain._fix_union_limit(sql), sql)


class TestSourceLabel(unittest.TestCase):
    def setUp(self):
        self.chain = _make_chain()

    def test_standalone_label(self):
        label = self.chain._suggest_source_label(
            {"source_file": "01深圳星辰数字科技集团股份有限公司_2025TB.xlsx"}
        )
        self.assertIn("(单体)", label)
        self.assertNotIn("01", label)
        self.assertNotIn(".xlsx", label)
        self.assertNotIn("2025TB", label)

    def test_merged_label(self):
        label = self.chain._suggest_source_label(
            {"source_file": "合并1：辰拓合并_2025TB.xlsx"}
        )
        self.assertIn("(合并)", label)


class TestCurrencyHelpers(unittest.TestCase):
    def setUp(self):
        self.chain = _make_chain()

    def test_strip_currency_pairs_match(self):
        a = self.chain._strip_currency("08-1STARNOVA MALAYSIA SDN. BHD._2025TB（人民币）.xlsx")
        b = self.chain._strip_currency("08-1STARNOVA MALAYSIA SDN. BHD._2025TB（林吉特）.xlsx")
        self.assertEqual(a, b)

    def test_strip_currency_no_currency_unchanged(self):
        f = "02深圳市辰拓智能设备有限公司_2025TB.xlsx"
        self.assertEqual(self.chain._strip_currency(f), f)


def _t(table_name, source_file, sheet_name="利润表"):
    return {"table_name": table_name, "source_file": source_file,
            "sheet_name": sheet_name, "row_count": 50, "columns": "项目,本期数"}


class TestAugmentCaliber(unittest.TestCase):
    """口径补全：未指明口径时自动补齐单体↔合并"""

    def setUp(self):
        self.chain = _make_chain()
        self.std = _t("t_hantuo_std", "02深圳市辰拓智能设备有限公司_2025TB.xlsx")
        self.mrg = _t("t_hantuo_mrg", "合并1：辰拓合并_2025TB.xlsx")
        self.all_tables = [self.std, self.mrg]

    def test_augment_adds_sibling_caliber(self):
        out = self.chain._augment_caliber("辰拓净利润是多少", [self.std], self.all_tables)
        names = {t["table_name"] for t in out}
        self.assertEqual(names, {"t_hantuo_std", "t_hantuo_mrg"})

    def test_explicit_standalone_not_augmented(self):
        out = self.chain._augment_caliber("辰拓单体净利润是多少", [self.std], self.all_tables)
        self.assertEqual(len(out), 1)

    def test_explicit_consolidated_not_augmented(self):
        out = self.chain._augment_caliber("辰拓合并口径净利润", [self.mrg], self.all_tables)
        self.assertEqual(len(out), 1)

    def test_different_sheet_not_augmented(self):
        other_sheet = _t("t_hantuo_mrg_bs", "合并1：辰拓合并_2025TB.xlsx", "资产负债表")
        out = self.chain._augment_caliber(
            "辰拓净利润是多少", [self.std], [self.std, other_sheet]
        )
        self.assertEqual(len(out), 1)

    def test_empty_selection_passthrough(self):
        self.assertEqual(self.chain._augment_caliber("辰拓净利润", [], self.all_tables), [])


class TestAugmentCurrency(unittest.TestCase):
    """币种补全：多币种版本未指定币种时全选"""

    def setUp(self):
        self.chain = _make_chain()
        self.rmb = _t("t_my_rmb", "08-1STARNOVA MALAYSIA SDN. BHD._2025TB（人民币）.xlsx")
        self.myr = _t("t_my_myr", "08-1STARNOVA MALAYSIA SDN. BHD._2025TB（林吉特）.xlsx")
        self.all_tables = [self.rmb, self.myr]

    def test_augment_adds_other_currency(self):
        out = self.chain._augment_currency("马来西亚星辰净利润", [self.rmb], self.all_tables)
        names = {t["table_name"] for t in out}
        self.assertEqual(names, {"t_my_rmb", "t_my_myr"})

    def test_explicit_currency_not_augmented(self):
        out = self.chain._augment_currency("马来西亚星辰人民币口径净利润", [self.rmb], self.all_tables)
        self.assertEqual(len(out), 1)

    def test_single_currency_file_not_augmented(self):
        std = _t("t_hantuo_std", "02深圳市辰拓智能设备有限公司_2025TB.xlsx")
        out = self.chain._augment_currency("辰拓净利润", [std], [std, self.rmb])
        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()

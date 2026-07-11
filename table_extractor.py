"""结构化数据层：表格清洗 + 字段标准化 + SQLite建模存储"""

import re
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from config import DB_PATH, MAX_SQL_ROWS
from document_processor import TableBlock


class TableExtractor:
    """表格数据清洗、标准化、存入SQLite"""

    TIME_PATTERNS = [
        (r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})", "%Y-%m-%d"),
        (r"(\d{4})[-/年](\d{1,2})[-/月]?", "%Y-%m"),
        (r"(\d{4})", "%Y"),
    ]

    AMOUNT_KEYWORDS = [
        "金额", "合计", "总计", "费用", "成本", "收入", "利润",
        "税额", "价税", "单价", "总价", "余额", "借方", "贷方",
        "应收", "应付", "预算", "实际", "差异",
    ]

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS table_registry (
                    table_name TEXT PRIMARY KEY,
                    source_file TEXT,
                    sheet_name TEXT,
                    page_num INTEGER,
                    columns TEXT,
                    row_count INTEGER,
                    created_at TEXT
                )
            """)

    def process_and_store(self, table_block: TableBlock) -> str:
        df = table_block.dataframe.copy()
        df = self._clean(df)
        df = self._standardize_fields(df)

        table_name = self._generate_table_name(table_block)

        with sqlite3.connect(self.db_path) as conn:
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            conn.execute("""
                INSERT OR REPLACE INTO table_registry
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                table_name,
                table_block.source_file,
                table_block.sheet_name,
                table_block.page_num,
                ",".join(df.columns.tolist()),
                len(df),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ))

        logger.info(f"表格存储: {table_name} ({len(df)}行, {len(df.columns)}列)")
        return table_name

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(how="all").dropna(axis=1, how="all")
        df.columns = [
            re.sub(r"\s+", "", str(c)).strip() if pd.notna(c) else f"列{i}"
            for i, c in enumerate(df.columns)
        ]
        seen = {}
        new_cols = []
        for c in df.columns:
            if c in seen:
                seen[c] += 1
                new_cols.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                new_cols.append(c)
        df.columns = new_cols
        return df.reset_index(drop=True)

    def _standardize_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if any(kw in col for kw in self.AMOUNT_KEYWORDS):
                converted = df[col].apply(self._parse_amount)
                # 防误伤：列名含金额关键词但实为文本列（如"收入类型""成本说明"），
                # 若原本有值却几乎全部无法解析为数字，判定为文本列，保留原样不转换
                orig_nonnull = df[col].notna().sum()
                conv_nonnull = converted.notna().sum()
                if orig_nonnull > 0 and conv_nonnull / orig_nonnull < 0.5:
                    continue
                df[col] = converted
            elif "日期" in col or "时间" in col or "date" in col.lower():
                df[col] = df[col].apply(self._parse_date)
        return df

    def _parse_amount(self, val) -> float | None:
        if pd.isna(val):
            return None
        s = str(val).replace(",", "").replace("，", "").replace(" ", "")
        s = re.sub(r"[元¥￥$]", "", s)
        try:
            return round(float(s), 2)
        except ValueError:
            return None

    def _parse_date(self, val) -> str | None:
        if pd.isna(val):
            return None
        s = str(val).strip()
        for pattern, _ in self.TIME_PATTERNS:
            m = re.search(pattern, s)
            if m:
                groups = m.groups()
                if len(groups) == 3:
                    return f"{groups[0]}-{int(groups[1]):02d}-{int(groups[2]):02d}"
                elif len(groups) == 2:
                    return f"{groups[0]}-{int(groups[1]):02d}"
                else:
                    return groups[0]
        return s

    def _generate_table_name(self, block: TableBlock) -> str:
        base = Path(block.source_file).stem
        base = re.sub(r"[^\w一-鿿]", "_", base)
        if block.sheet_name:
            sheet = re.sub(r"[^\w一-鿿]", "_", block.sheet_name)
            return f"t_{base}_{sheet}"[:60]
        if block.page_num:
            return f"t_{base}_p{block.page_num}_t{block.table_index}"[:60]
        return f"t_{base}"[:60]

    def get_all_tables_info(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM table_registry").fetchall()
        return [
            {
                "table_name": r[0], "source_file": r[1], "sheet_name": r[2],
                "page_num": r[3], "columns": r[4], "row_count": r[5], "created_at": r[6],
            }
            for r in rows
        ]

    def delete_by_source(self, source_file: str):
        with sqlite3.connect(self.db_path) as conn:
            tables = conn.execute(
                "SELECT table_name FROM table_registry WHERE source_file = ?", (source_file,)
            ).fetchall()
            for (tname,) in tables:
                conn.execute(f'DROP TABLE IF EXISTS "{tname}"')
            conn.execute("DELETE FROM table_registry WHERE source_file = ?", (source_file,))
        logger.info(f"已删除来源表格: {source_file} ({len(tables)}张表)")

    def get_source_files(self) -> list[str]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT DISTINCT source_file FROM table_registry").fetchall()
        return [r[0] for r in rows]

    def query(self, sql: str) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(sql, conn)

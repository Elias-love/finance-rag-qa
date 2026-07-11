"""重建索引脚本：清空SQLite + ChromaDB，重新解析uploads目录下所有文件"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import DB_PATH, CHROMA_DIR, UPLOAD_DIR
from document_processor import DocumentProcessor
from table_extractor import TableExtractor
from vector_store import VectorStore


def clear_sqlite():
    """清空SQLite的所有数据表"""
    if not DB_PATH.exists():
        print("SQLite 数据库不存在，跳过")
        return
    conn = sqlite3.connect(DB_PATH)
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    for (tname,) in tables:
        conn.execute(f'DROP TABLE IF EXISTS "{tname}"')
    conn.commit()
    conn.close()
    print(f"✅ 已清空 SQLite ({len(tables)} 张表)")


def clear_chroma():
    """清空ChromaDB集合"""
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection("finance_docs")
        print("✅ 已清空 ChromaDB 集合")
    except Exception as e:
        print(f"⚠️  ChromaDB 清空跳过: {e}")


def rebuild():
    print("=" * 60)
    print("【重建知识库索引】")
    print("=" * 60)

    # 1. 清空
    clear_sqlite()
    clear_chroma()

    # 2. 列出所有上传文件
    files = sorted(UPLOAD_DIR.glob("*"))
    files = [f for f in files if f.is_file() and not f.name.startswith(".")]
    print(f"\n找到 {len(files)} 个待重建文件")

    if not files:
        print("无文件可重建")
        return

    # 3. 重新初始化组件
    processor = DocumentProcessor()
    table_ext = TableExtractor()
    vector_store = VectorStore()

    # 4. 逐个重新解析入库
    total_tables = 0
    total_chunks = 0
    failed = []

    for i, f in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {f.name}")
        try:
            result = processor.parse(f)

            chunks = 0
            if result.texts:
                chunks = vector_store.add_texts(result.texts)

            tables = 0
            for tb in result.tables:
                table_ext.process_and_store(tb)
                tables += 1

            total_chunks += chunks
            total_tables += tables
            print(f"  → 文本块{chunks} | 数据表{tables}")
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            failed.append((f.name, str(e)))

    # 5. 汇总
    print("\n" + "=" * 60)
    print(f"重建完成: {len(files) - len(failed)}/{len(files)} 文件成功")
    print(f"总文本块: {total_chunks}")
    print(f"总数据表: {total_tables}")
    if failed:
        print(f"\n失败列表:")
        for name, err in failed:
            print(f"  - {name}: {err}")
    print("=" * 60)


if __name__ == "__main__":
    rebuild()

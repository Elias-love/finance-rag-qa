"""PDF嵌入图片的存储与按需检索"""

import sqlite3
from pathlib import Path

from loguru import logger

from config import DB_PATH
from document_processor import ImageBlock


class ImageStore:
    """记录PDF中提取的图片，支持按文件+页码反查"""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS page_images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_file TEXT NOT NULL,
                    page_num INTEGER NOT NULL,
                    image_index INTEGER,
                    image_path TEXT NOT NULL,
                    width INTEGER,
                    height INTEGER,
                    caption TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_page_images_lookup "
                "ON page_images(source_file, page_num)"
            )

    def add_images(self, images: list[ImageBlock]) -> int:
        if not images:
            return 0
        with sqlite3.connect(self.db_path) as conn:
            for img in images:
                conn.execute("""
                    INSERT INTO page_images
                    (source_file, page_num, image_index, image_path, width, height, caption)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    img.source_file, img.page_num, img.image_index,
                    img.image_path, img.width, img.height, img.caption,
                ))
        logger.info(f"图片索引入库: {len(images)}张")
        return len(images)

    def get_images_by_pages(self, source_file: str, pages: list[int]) -> list[dict]:
        """按文件+页码列表查询图片"""
        if not pages:
            return []
        with sqlite3.connect(self.db_path) as conn:
            placeholders = ",".join("?" * len(pages))
            rows = conn.execute(
                f"SELECT source_file, page_num, image_index, image_path, width, height "
                f"FROM page_images WHERE source_file = ? AND page_num IN ({placeholders}) "
                f"ORDER BY page_num, image_index",
                [source_file] + pages,
            ).fetchall()
        return [
            {
                "source_file": r[0], "page_num": r[1], "image_index": r[2],
                "image_path": r[3], "width": r[4], "height": r[5],
            }
            for r in rows
        ]

    def get_images_for_sources(self, sources: list[dict]) -> list[dict]:
        """根据RAG引用来源列表(file+page)批量查询图片"""
        # 按 source_file 分组
        groups = {}
        for s in sources:
            if not s.get("file") or not s.get("page"):
                continue
            groups.setdefault(s["file"], set()).add(s["page"])

        all_images = []
        for fname, pages in groups.items():
            imgs = self.get_images_by_pages(fname, list(pages))
            all_images.extend(imgs)
        return all_images

    def delete_by_source(self, source_file: str):
        """删除某个文件的所有图片记录及本地文件"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT image_path FROM page_images WHERE source_file = ?",
                (source_file,)
            ).fetchall()
            for (path,) in rows:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass
            conn.execute("DELETE FROM page_images WHERE source_file = ?", (source_file,))
        logger.info(f"已删除图片索引: {source_file} ({len(rows)}张)")

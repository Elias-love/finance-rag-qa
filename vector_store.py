"""文本知识层：语义切片 + ChromaDB + 丰富元数据（页码/章节/原文）"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import hashlib
import re
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from loguru import logger

from config import CHROMA_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from document_processor import TextBlock


class SemanticChunker:
    """语义感知切片：按标题/段落边界切分，而非固定字数"""

    def __init__(self, max_chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.max_size = max_chunk_size
        self.overlap = overlap

    def chunk_blocks(self, blocks: list[TextBlock]) -> list[dict]:
        """将结构化文本块切成带元数据的chunk"""
        chunks = []

        # 按 (source_file, page_num, section) 分组
        groups = self._group_blocks(blocks)

        for group_key, group_blocks in groups.items():
            source_file, page_num, section = group_key

            # 合并同组文本
            merged_text = ""
            heading_path = ""
            block_types = set()

            for b in group_blocks:
                if b.block_type == "heading":
                    merged_text += f"\n\n{b.content}\n"
                elif b.block_type == "footnote":
                    merged_text += f"\n[注]{b.content}"
                else:
                    merged_text += f"\n{b.content}"

                if b.heading_path:
                    heading_path = b.heading_path
                block_types.add(b.block_type)

            merged_text = merged_text.strip()
            if len(merged_text) < 10:
                continue

            # 如果文本短于阈值，直接作为一个chunk
            if len(merged_text) <= self.max_size:
                chunks.append({
                    "text": merged_text,
                    "source_file": source_file,
                    "page_num": page_num,
                    "section": section,
                    "heading_path": heading_path,
                    "block_type": ",".join(sorted(block_types)),
                    "chunk_index": 0,
                })
            else:
                # 按语义边界切分
                sub_chunks = self._split_by_semantics(merged_text)
                for i, sc in enumerate(sub_chunks):
                    chunks.append({
                        "text": sc,
                        "source_file": source_file,
                        "page_num": page_num,
                        "section": section,
                        "heading_path": heading_path,
                        "block_type": ",".join(sorted(block_types)),
                        "chunk_index": i,
                    })

        return chunks

    def _group_blocks(self, blocks: list[TextBlock]) -> dict:
        """按 (文件, 页码, 章节) 分组"""
        groups = {}
        for b in blocks:
            # 跳过目录类型
            if b.block_type == "toc":
                continue
            key = (b.source_file, b.page_num, b.section)
            groups.setdefault(key, []).append(b)
        return groups

    def _split_by_semantics(self, text: str) -> list[str]:
        """按段落/句子边界切分，保持语义完整"""
        # 先按段落分
        paragraphs = re.split(r"\n{2,}", text)

        chunks = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current) + len(para) + 1 <= self.max_size:
                current = f"{current}\n{para}" if current else para
            else:
                if current:
                    chunks.append(current.strip())
                # 如果单个段落超长，按句子切分
                if len(para) > self.max_size:
                    sentences = re.split(r"(?<=[。！？；\.\!\?\;])", para)
                    sub_current = ""
                    for sent in sentences:
                        sent = sent.strip()
                        if not sent:
                            continue
                        if len(sub_current) + len(sent) <= self.max_size:
                            sub_current = f"{sub_current}{sent}" if sub_current else sent
                        else:
                            if sub_current:
                                chunks.append(sub_current.strip())
                            sub_current = sent
                    if sub_current:
                        current = sub_current
                    else:
                        current = ""
                else:
                    current = para

        if current:
            chunks.append(current.strip())

        # 添加重叠
        if len(chunks) > 1 and self.overlap > 0:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                prev_tail = chunks[i - 1][-self.overlap:]
                overlapped.append(f"{prev_tail}…\n{chunks[i]}")
            chunks = overlapped

        return [c for c in chunks if len(c) >= 10]


class VectorStore:
    """ChromaDB 文本向量存储与检索，支持丰富元数据"""

    def __init__(self):
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
            trust_remote_code=False,
        )
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(
            name="finance_docs",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self.chunker = SemanticChunker()
        # 混合召回器懒加载：首次 search_hybrid 时构建 BM25 索引
        self._hybrid = None

    @property
    def hybrid(self):
        if self._hybrid is None:
            from hybrid_retriever import HybridRetriever
            self._hybrid = HybridRetriever(self.collection, self.embedding_fn)
        return self._hybrid

    def search_hybrid(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """向量+BM25双路召回 → RRF融合 → 可选重排。返回结构兼容 search()。"""
        return self.hybrid.search(query, top_k=top_k)

    def add_texts(self, text_blocks: list[TextBlock]) -> int:
        chunks = self.chunker.chunk_blocks(text_blocks)

        if not chunks:
            return 0

        documents, metadatas, ids = [], [], []

        for chunk in chunks:
            text = chunk["text"]
            doc_id = hashlib.md5(
                f"{chunk['source_file']}:{chunk['page_num']}:{chunk['chunk_index']}:{text[:50]}".encode()
            ).hexdigest()

            documents.append(text)
            ids.append(doc_id)
            metadatas.append({
                "source_file": chunk["source_file"],
                "page_num": chunk["page_num"],
                "section": chunk["section"],
                "heading_path": chunk["heading_path"],
                "block_type": chunk["block_type"],
                "chunk_index": chunk["chunk_index"],
                "char_count": len(text),
            })

        batch_size = 100
        for start in range(0, len(documents), batch_size):
            end = start + batch_size
            self.collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

        logger.info(f"向量化完成: {len(documents)}个语义块")
        return len(documents)

    def search(self, query: str, top_k: int = TOP_K,
               filter_dict: dict | None = None) -> list[dict]:
        kwargs = {
            "query_texts": [query],
            "n_results": top_k,
        }
        if filter_dict:
            kwargs["where"] = filter_dict

        results = self.collection.query(**kwargs)

        hits = []
        for i in range(len(results["documents"][0])):
            hits.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return hits

    def get_stats(self) -> dict:
        return {"total_chunks": self.collection.count()}

    def get_source_files(self) -> list[str]:
        """文本知识库中的文档清单（供意图路由判断问题是否可被文档回答）"""
        if self.collection.count() == 0:
            return []
        data = self.collection.get(include=["metadatas"])
        return sorted({m["source_file"] for m in data["metadatas"] if m.get("source_file")})

    def delete_by_source(self, source_file: str):
        self.collection.delete(where={"source_file": source_file})
        logger.info(f"已删除来源: {source_file}")

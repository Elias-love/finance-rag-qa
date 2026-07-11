"""安全与治理层：SQL白名单 + 敏感数据脱敏 + 查询日志 + 来源追踪"""

import re
import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from config import SENSITIVE_FIELDS, LOG_DIR


class SecurityManager:
    """安全管控"""

    # 日志滚动上限：超过则只保留最近 MAX_LOG_LINES 条，防止文件无限增长
    MAX_LOG_LINES = 5000

    def __init__(self):
        self.log_file = LOG_DIR / "query_log.jsonl"

    def mask_sensitive(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        # 身份证：18位，最后一位可能为X
        text = re.sub(r"(?<![\d.])\d{17}[\dXx](?![\d.])", "****身份证****", text)
        # 银行卡①：带"卡号/银行卡/账号"上下文的 13-19 位（覆盖16位信用卡）
        text = re.sub(r"((?:卡号|银行卡|账号)[：:\s]*)\d{13,19}",
                      r"\1****银行卡****", text)
        # 银行卡②：独立的 19 位储蓄卡（不匹配16-18位，避免误伤16位订单号/流水号）
        text = re.sub(r"(?<![\d.,])\d{19}(?![\d.,])", "****银行卡****", text)
        # 手机号：1开头11位
        text = re.sub(r"(?<![\d.])1[3-9]\d{9}(?![\d.])", "****手机****", text)
        return text

    def mask_dataframe(self, df):
        """对表格脱敏：仅处理文本列，数值列(金额)原样保留避免误伤"""
        if df is None or getattr(df, "empty", True):
            return df
        df = df.copy()
        import pandas as pd
        for col in df.columns:
            # pandas 3.0 起字符串列 dtype 为 str（不再是 object），两种都要覆盖
            if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].apply(
                    lambda v: self.mask_sensitive(v) if isinstance(v, str) else v
                )
        return df

    def check_question_safety(self, question: str) -> dict:
        for field in SENSITIVE_FIELDS:
            if field in question:
                return {
                    "safe": False,
                    "warning": f"问题中包含敏感字段「{field}」，请确认是否需要查询此类信息。",
                }
        return {"safe": True, "warning": ""}

    def log_query(self, question: str, intent: str, answer: str, sources: list,
                  user_id: str = "default", success: bool = True,
                  sql: str = "", latency_ms: int = 0):
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": user_id,
            "question": self.mask_sensitive(question),
            "intent": intent,
            "success": success,
            "latency_ms": latency_ms,
            "sql": sql[:500] if sql else "",
            "answer_length": len(answer),
            "source_count": len(sources),
            "sources": [s.get("file", s.get("type", "")) for s in sources] if sources else [],
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._rotate_if_needed()

    def log_feedback(self, question: str, answer: str, rating: str, user_id: str = "default"):
        """记录用户对回答的评价：rating ∈ {up, down}。用于事后评估和坏例回收。"""
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": user_id,
            "rating": rating,
            "question": self.mask_sensitive(question),
            "answer_head": self.mask_sensitive(answer[:200]),
        }
        feedback_file = LOG_DIR / "feedback_log.jsonl"
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _rotate_if_needed(self):
        """审计日志滚动：活动文件只保留最近 MAX_LOG_LINES 条，
        溢出的旧条目**按月归档到 logs/archive/ 而非删除**，满足审计长期留存要求。"""
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) <= self.MAX_LOG_LINES:
                return
            overflow = lines[:-self.MAX_LOG_LINES]
            recent = lines[-self.MAX_LOG_LINES:]
            archive_dir = LOG_DIR / "archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m")
            archive_file = archive_dir / f"query_log_{stamp}.jsonl"
            with open(archive_file, "a", encoding="utf-8") as f:
                f.writelines(overflow)   # 追加归档，绝不丢弃
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.writelines(recent)
            logger.info(
                f"审计日志滚动：{len(overflow)} 条已归档至 {archive_file.name}，"
                f"活动文件保留最近 {self.MAX_LOG_LINES} 条"
            )
        except Exception as e:
            logger.warning(f"日志滚动失败: {e}")

    def get_query_history(self, limit: int = 50) -> list[dict]:
        if not self.log_file.exists():
            return []
        # 只读尾部 limit 行，且逐行容错（坏行跳过），避免单条损坏拖垮整体
        from collections import deque
        with open(self.log_file, "r", encoding="utf-8") as f:
            tail = deque(f, maxlen=limit)
        out = []
        for line in tail:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

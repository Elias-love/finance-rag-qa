"""问答编排层：意图识别 + 选表（合并为一次LLM调用）"""

import json
from openai import OpenAI
from loguru import logger

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


COMBINED_PROMPT = """你是财务问答系统的智能路由器。任务：
（1）判定用户问题的意图类型
（2）如果是data类型，从候选表清单中选出最相关的1-5张表

【意图类型】
1. **data** — 涉及任何报表数字的查询、汇总、对比、解读、推导、原因分析
   只要问题提到具体报表项目（净利润/营业收入/资产/负债等）或公司名，强制归为 data。
   不要因为用户问"为什么/怎么算"就误判为 text，这些是对 data 的二次解读。

2. **text** — 查询纯文字制度、规定、流程、条款、准则（与具体报表数字无关）
   例: 「报销流程是什么」「固定资产折旧政策的会计准则」
   **注意**：考勤、作息、休假、用餐时间、办公规范等行政制度类问题也归 text。
   用户可能用生活化口吻提问（如「中午12点可以去吃饭吗」「几点下班」「可以调休半天吗」），
   只要【文本知识库文档清单】中有可能覆盖该话题的文档（如考勤制度、工作手册），就归为 text。

3. **hybrid** — 同时涉及数据和制度文本

4. **clarify** — 问题与数据表、文本文档都无法关联，模糊到完全无法判断时才用。
   能对应到任一文档主题的问题不要归 clarify，宁可归 text 让检索层判断。

【选表规则（仅 data 类型需要）】
- 口径概念：文件名含"合并"的是合并报表；其他是单体报表。
- "深圳星辰数字科技集团股份有限公司"是上市主体的母公司单体，"合并0：星辰集团"是合并口径。
- 用户提到具体公司名 → 选该公司的对应表。
- 用户提到"集团"或集团名（如"星辰集团"）但**未指明口径** → 同时选母公司单体表和集团合并表两张。
- 用户明确说"合并" → 只选合并表；明确说"母公司/单体/单家" → 只选单体表。
- 同一报表类型在不同公司都相关时，可多选。

【输出 JSON 格式】
{
  "intent": "data" | "text" | "hybrid" | "clarify",
  "tables": ["表名1", "表名2"]   // 仅 data 类型需要填，其他类型留空数组 []
}"""


class QueryRouter:
    def __init__(self):
        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    def classify_and_select(self, question: str, table_list_str: str = "",
                             table_count: int = 0, text_docs: list[str] = None) -> dict:
        """合并意图分类 + 选表，一次 LLM 调用完成"""
        context = (
            f"候选表清单（{table_count}张）:\n{table_list_str}"
            if table_list_str else "数据库中暂无表"
        )
        if text_docs:
            context += "\n\n文本知识库文档清单:\n" + "\n".join(f"- {d}" for d in text_docs)

        response = self.client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": COMBINED_PROMPT},
                {"role": "user", "content": f"{context}\n\n用户问题：{question}"},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        result.setdefault("intent", "clarify")
        result.setdefault("tables", [])
        logger.info(f"路由+选表: '{question[:30]}...' → {result['intent']}, 选中{len(result['tables'])}张")
        return result

    # 兼容旧接口
    def classify(self, question: str, table_info: str = "") -> dict:
        r = self.classify_and_select(question, table_info)
        return {"intent": r["intent"], "tables": r["tables"]}

"""公司名同音纠错：用户把公司名打成同音错别字时（如"兴美"→"星美"），
找出最可能的正确公司，供前端"提示候选 + 用户确认"，绝不静默替换。"""

import re
from pypinyin import lazy_pinyin

from sql_chain import COMPANY_ALIASES

# 公司简称 → 规范全名。简称同时是 sql_chain 预筛能命中的关键词，
# 因此用户确认后把错字替换成简称即可正常查询。
COMPANY_DIRECTORY = {
    "星美": "深圳星美智能材料有限公司",
    "星博": "深圳星博智能装备有限公司",
    "星源": "广东星源智能科技有限公司",
    "星锐": "深圳星锐精密装备有限公司",
    "辰拓": "深圳市辰拓智能设备有限公司",
    "诚跃": "深圳诚跃智能设备有限公司",
    "辰华": "深圳辰华工业设备有限公司",
    "佳星": "广东佳星新材料科技有限公司",
    "东晟": "珠海东晟新材料科技有限公司",
    "惠州": "惠州星辰实业有限公司",
    "江苏": "江苏华辰智能装备有限公司",
}

# 预计算简称拼音（不带声调，便于同音匹配）
_NAME_PY = {name: "".join(lazy_pinyin(name)) for name in COMPANY_DIRECTORY}

# 已知可被精确识别的公司词（输对了就不该触发纠错）
_KNOWN_WORDS = set(COMPANY_DIRECTORY) | set(COMPANY_ALIASES) | {
    "星辰", "星博", "星辰软件", "星辰集团", "集团", "合并",
}


def detect_company_typo(question: str) -> list[dict]:
    """检测问题里所有可能的公司名同音错别字（可能多个）。

    返回 list[{wrong, right, name}]：用户问"星播和兴瑞2025净利润"会一次返回
    [{wrong:星播,right:星博,...}, {wrong:兴瑞,right:星锐,...}]。
    无可疑错字（输对了 / 没提公司 / 制度类问题）时返回 []。
    """
    if not question:
        return []

    # ① 移除问题中已精确命中的已知公司词，避免它们参与滑窗（防误纠正确写法）
    masked = question
    for w in _KNOWN_WORDS:
        if w in masked:
            masked = masked.replace(w, "　")  # 用全角空格隔断滑窗

    fixes = []
    seen_wrong = set()  # 同一错词只提示一次（多个滑窗会重复命中）

    # ② 对剩余中文片段做 2~4 字滑窗，找"与某公司简称完全同音但字不同"的词
    for seg in re.findall(r"[一-鿿]{2,}", masked):
        for w in range(2, 5):
            for i in range(len(seg) - w + 1):
                sub = seg[i:i + w]
                if sub in seen_wrong:
                    continue
                sub_py = "".join(lazy_pinyin(sub))
                for name, npy in _NAME_PY.items():
                    if sub_py == npy and sub != name:
                        fixes.append({
                            "wrong": sub,
                            "right": name,
                            "name": COMPANY_DIRECTORY[name],
                        })
                        seen_wrong.add(sub)
                        break  # 已命中一个候选公司，无需再比其他
    return fixes

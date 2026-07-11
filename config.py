import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
UPLOAD_DIR = DATA_DIR / "uploads"
EXPORT_DIR = DATA_DIR / "exports"
DB_PATH = DATA_DIR / "finance.db"
LOG_DIR = BASE_DIR / "logs"

for d in [CHROMA_DIR, UPLOAD_DIR, EXPORT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# 管理员账号（从环境变量读取，避免硬编码在源码）
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "finance2025")
# 登录安全：失败锁定阈值与会话超时（秒）
LOGIN_MAX_FAILS = int(os.getenv("LOGIN_MAX_FAILS", "5"))
LOGIN_LOCK_SECONDS = int(os.getenv("LOGIN_LOCK_SECONDS", "300"))
SESSION_TIMEOUT_SECONDS = int(os.getenv("SESSION_TIMEOUT_SECONDS", "1800"))
# 全站访问口令：留空=开放（本地测试）；设值=部署时任何人问答前须先通过口令
# 部署到服务器/内网时务必设置，否则财务数据对所有能访问端口的人裸露
APP_ACCESS_PASSWORD = os.getenv("APP_ACCESS_PASSWORD", "")

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 5

# ===== 混合召回参数 =====
# 召回阶段宽（向量 + BM25 各取 RECALL_K），融合/重排后收窄到 TOP_K
RECALL_K = int(os.getenv("RECALL_K", "20"))
RRF_K = 60                     # RRF 融合常数，业界标准值
# 向量距离门槛：距离(=1-余弦相似度)低于此值才视为相关。
# 2026-07-04 由 evaluate.py --sweep 在黄金集(21正例+3负例)上标定：
# 0.50 答对23/24。0.45 会挡掉生活化问法（"中午12点可以去吃饭吗"距离0.488），
# 0.55 则3道负例全部误召回。改动前先重跑 --sweep
VEC_DISTANCE_GATE = float(os.getenv("VEC_DISTANCE_GATE", "0.40"))
VEC_RELATIVE_WINDOW = 0.12     # 与最佳命中的距离差窗口，剔除掉队结果
# 重排序（可选）：模型已本地缓存时自动启用；设 RERANK_ENABLED=0 强制关闭
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "auto")   # auto / 1 / 0
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0.3"))

SQL_ALLOWED_OPS = {"SELECT"}
SQL_BLOCKED_OPS = {"DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE"}
MAX_SQL_ROWS = 1000

SENSITIVE_FIELDS = ["身份证", "银行卡", "手机号", "密码", "社保号"]

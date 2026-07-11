"""财务RAG问答系统 — 问答免登录，文档管理需管理员认证"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import hashlib
import hmac
import threading
import time
import streamlit as st
import pandas as pd
from pathlib import Path
from loguru import logger

import chart_helper
import company_matcher

from config import (
    UPLOAD_DIR, EXPORT_DIR,
    ADMIN_USER, ADMIN_PASSWORD, APP_ACCESS_PASSWORD,
    LOGIN_MAX_FAILS, LOGIN_LOCK_SECONDS, SESSION_TIMEOUT_SECONDS,
)

st.set_page_config(page_title="财务知识库问答系统", page_icon="📊", layout="wide")

# ============================================================
# 长时间空闲保活：浏览器端定期心跳 + WebSocket断连自动重连
# ============================================================
st.markdown("""
<script>
(function() {
    if (window.__finance_keepalive_installed__) return;
    window.__finance_keepalive_installed__ = true;

    // 每60秒触发一次轻量交互，避免 Streamlit WebSocket 因空闲被中间代理切断
    setInterval(function () {
        try { window.dispatchEvent(new Event('mousemove')); } catch(e) {}
    }, 60000);

    // 监听连接状态：失联超过3秒自动刷新（保留URL）
    let disconnectedSince = null;
    setInterval(function () {
        try {
            const overlay = document.querySelector('[data-testid="stConnectionStatus"]');
            const overlayText = overlay ? overlay.innerText : "";
            const isDisconnected = /Connecting|Disconnected|断开|重新连接/i.test(overlayText);

            if (isDisconnected) {
                if (!disconnectedSince) disconnectedSince = Date.now();
                if (Date.now() - disconnectedSince > 3000) {
                    location.reload();
                }
            } else {
                disconnectedSince = null;
            }
        } catch(e) {}
    }, 2000);

    // 标签页切回前台时主动刷一次，避免后台tab的状态错乱
    document.addEventListener('visibilitychange', function () {
        if (document.visibilityState === 'visible') {
            const last = parseInt(sessionStorage.getItem('lastSeen') || '0');
            if (last && Date.now() - last > 5 * 60 * 1000) {
                location.reload();
            }
        }
        sessionStorage.setItem('lastSeen', Date.now());
    });
    sessionStorage.setItem('lastSeen', Date.now());
})();
</script>
""", unsafe_allow_html=True)

# ============================================================
# 管理员账号（仅管理文档时需要）
# ============================================================
ADMINS = {
    ADMIN_USER: hashlib.sha256(ADMIN_PASSWORD.encode()).hexdigest(),
}

# ============================================================
# 组件初始化（带进度条）
# ============================================================
INIT_STEPS = [
    (10, "加载文档解析器"),
    (25, "加载表格抽取器"),
    (45, "加载向量库与嵌入模型"),
    (70, "加载问答编排器"),
    (85, "加载导出器"),
    (95, "加载安全模块"),
    (100, "初始化完成"),
]


def _code_version() -> str:
    """所有 .py 文件的 mtime 指纹。作为 cache_resource 的键，
    代码更新后自动重建组件，避免缓存的旧实例与热重载的新代码不匹配
    （否则会出现 log_query() got an unexpected keyword argument 这类错误）"""
    import hashlib
    files = sorted(Path(__file__).parent.glob("*.py"))
    stamp = "|".join(f"{f.name}:{f.stat().st_mtime_ns}" for f in files)
    return hashlib.md5(stamp.encode()).hexdigest()[:12]


@st.cache_resource(show_spinner=False)
def _build_components(code_version: str = ""):
    """实际构建组件，不在内部调用任何streamlit UI元素（避免cache replay错误）"""
    from document_processor import DocumentProcessor
    from table_extractor import TableExtractor
    from vector_store import VectorStore
    from orchestrator import Orchestrator
    from exporter import Exporter
    from security import SecurityManager
    return {
        "processor": DocumentProcessor(),
        "table_ext": TableExtractor(),
        "vector_store": VectorStore(),
        "orchestrator": Orchestrator(),
        "exporter": Exporter(),
        "security": SecurityManager(),
    }


def init_components():
    """首屏带进度条；命中缓存时秒过"""
    if st.session_state.get("_components_loaded"):
        return _build_components(_code_version())

    placeholder = st.empty()
    bar = placeholder.progress(0, text="正在加载知识库系统，请稍候…")
    for pct, msg in INIT_STEPS[:-1]:
        bar.progress(pct, text=f"正在加载知识库系统：{msg}（{pct}%）")
    comps = _build_components(_code_version())
    bar.progress(100, text="加载完成")
    placeholder.empty()
    st.session_state._components_loaded = True
    return comps


components = init_components()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_data" not in st.session_state:
    st.session_state.last_data = None
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False
if "login_time" not in st.session_state:
    st.session_state.login_time = 0.0
if "session_uid" not in st.session_state:
    # 会话级审计标识：接入 SSO 前先做到"可追溯到会话"，优于恒为 default
    import uuid
    st.session_state.session_uid = uuid.uuid4().hex[:12]


def _current_user_id() -> str:
    """审计用身份：管理员登录时用其用户名，否则用匿名会话短ID。
    （接入企业 SSO 后，把此处换成 SSO 返回的真实工号即可全链路生效）"""
    if st.session_state.get("admin_logged_in"):
        return f"admin:{ADMIN_USER}"
    return f"anon:{st.session_state.get('session_uid', 'unknown')}"
if "uploader_seq" not in st.session_state:
    st.session_state.uploader_seq = 0
if "ingest_result" not in st.session_state:
    st.session_state.ingest_result = None
if "pending_confirm" not in st.session_state:
    st.session_state.pending_confirm = None
if "feedback" not in st.session_state:
    st.session_state.feedback = {}   # {消息索引: "up"/"down"}，防止重复提交


@st.cache_resource(show_spinner=False)
def _login_guard() -> dict:
    """跨会话共享的登录失败计数（服务端级）。
    修复：原来把失败次数存 st.session_state，攻击者换浏览器会话/无痕窗口即可
    重置计数绕过锁定。改到进程级共享后，锁定按用户名/闸门生效，换会话也无法绕过。
    结构：{"fails": {key: n}, "lock_until": {key: ts}, "lock": Lock}
    lock：并发多会话同时失败时，get+1 再回写非原子会丢增量，用锁保护读改写。"""
    return {"fails": {}, "lock_until": {}, "lock": threading.Lock()}


# ============================================================
# 全站访问口令闸门：设置了 APP_ACCESS_PASSWORD 时，任何人问答前须先通过口令。
# 未设置（本地测试）时完全开放，不改变现有体验。
# ============================================================
if APP_ACCESS_PASSWORD:
    st.session_state.setdefault("access_granted", False)
    if not st.session_state.access_granted:
        guard = _login_guard()
        now = time.time()
        lock_until = guard["lock_until"].get("__access__", 0.0)
        st.title("🔒 财务知识库")
        st.caption("本系统包含公司敏感财务数据，请输入访问口令后使用")
        if now < lock_until:
            st.error(f"🔒 尝试次数过多，请 {int(lock_until - now)} 秒后重试")
            st.stop()
        _pwd = st.text_input("访问口令", type="password", key="access_pwd")
        if st.button("进入", key="access_enter_btn"):
            # encode 成 bytes：口令含非 ASCII 字符时 compare_digest 直接比 str 会抛 TypeError
            if hmac.compare_digest(_pwd.encode(), APP_ACCESS_PASSWORD.encode()):
                with guard["lock"]:
                    guard["fails"].pop("__access__", None)
                st.session_state.access_granted = True
                st.rerun()
            else:
                with guard["lock"]:   # 读改写加锁，防并发丢增量削弱锁定
                    fails = guard["fails"].get("__access__", 0) + 1
                    guard["fails"]["__access__"] = fails
                    locked_now = fails >= LOGIN_MAX_FAILS
                    if locked_now:
                        guard["lock_until"]["__access__"] = now + LOGIN_LOCK_SECONDS
                        guard["fails"]["__access__"] = 0
                if locked_now:
                    st.error(f"🔒 尝试次数过多，锁定 {LOGIN_LOCK_SECONDS // 60} 分钟")
                else:
                    st.error(f"口令错误（还可尝试 {LOGIN_MAX_FAILS - fails} 次）")
        st.stop()


def _get_all_source_files() -> list[str]:
    chroma_files = set()
    try:
        all_data = components["vector_store"].collection.get()
        if all_data and all_data.get("metadatas"):
            for m in all_data["metadatas"]:
                if m.get("source_file"):
                    chroma_files.add(m["source_file"])
    except Exception:
        pass
    sqlite_files = set(components["table_ext"].get_source_files())
    return sorted(chroma_files | sqlite_files)


def _delete_source(filename: str):
    components["vector_store"].delete_by_source(filename)
    components["table_ext"].delete_by_source(filename)
    components["orchestrator"].image_store.delete_by_source(filename)
    # 路径穿越防护：只允许删除 uploads 目录内的文件
    try:
        local_file = _safe_upload_path(filename)
    except ValueError:
        logger.warning(f"拒绝删除非法路径文件: {filename}")
        return
    if local_file.exists():
        local_file.unlink()
    # 原始文件已变更：清空预览/下载的 cache_data（key 只含文件名，不会随内容失效），
    # 否则覆盖替换/删除后仍会提供旧版本字节
    st.cache_data.clear()


def _check_excel_formula_cache(uploaded_files) -> list[dict]:
    """检查上传的Excel文件是否含未重算的公式缓存，返回有风险的文件列表"""
    import openpyxl
    import tempfile
    warnings = []
    for uf in uploaded_files:
        if not uf.name.lower().endswith((".xlsx", ".xlsm")):
            continue
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=uf.name, delete=False)
            tmp.write(uf.getvalue())
            tmp.close()
            wb = openpyxl.load_workbook(tmp.name, read_only=True, data_only=False)
            calc = getattr(wb, "calculation", None)
            full_calc = getattr(calc, "fullCalcOnLoad", False) if calc else False
            formula_count = 0
            if full_calc:
                for ws in wb.worksheets:
                    for row in ws.iter_rows():
                        for cell in row:
                            if isinstance(cell.value, str) and cell.value.startswith("="):
                                formula_count += 1
                                if formula_count >= 10:
                                    break
                        if formula_count >= 10:
                            break
                    if formula_count >= 10:
                        break
            wb.close()
            os.unlink(tmp.name)
            if full_calc and formula_count >= 10:
                warnings.append({"name": uf.name, "formulas": formula_count})
            uf.seek(0)
        except Exception:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
            uf.seek(0)
    return warnings


import re as _re

_KNOWN_COMPANIES = [
    "星辰", "辰拓", "星美", "星博", "佳星", "星源", "诚跃", "星锐", "辰华",
    "东晟", "惠州", "江苏", "星辰软件", "合并",
    "STARNOVA", "STARNOVA", "Starnova", "MALAYSIA", "Graphics",
]

def _check_filename_quality(uploaded_files) -> list[dict]:
    """检查文件名是否包含可识别的公司名和年份，返回有问题的文件列表"""
    issues = []
    for uf in uploaded_files:
        name = uf.name
        stem = name.rsplit(".", 1)[0] if "." in name else name
        problems = []
        has_company = any(kw in stem for kw in _KNOWN_COMPANIES)
        if not has_company:
            problems.append("缺少公司名")
        has_year = bool(_re.search(r"20[0-9]{2}", stem))
        if not has_year:
            problems.append("缺少年份")
        if problems:
            issues.append({"name": name, "problems": problems})
    return issues


def _ingest_file(uf) -> str:
    """入库单个文件。失败时隔离处理，返回以 ❌ 开头的错误消息而不抛异常。"""
    # 路径穿越防护：阻断 uf.name 含 ../ 或绝对路径写到目录外
    try:
        save_path = _safe_upload_path(uf.name)
    except ValueError:
        return f"❌ {uf.name}: 文件名非法（含路径分隔符），已拒绝入库"
    try:
        save_path.write_bytes(uf.getvalue())
        st.cache_data.clear()  # 同名覆盖写入后，预览/下载缓存必须失效
        result = components["processor"].parse(save_path)
        text_count = 0
        if result.texts:
            text_count = components["vector_store"].add_texts(result.texts)
        table_names = []
        for tb in result.tables:
            tname = components["table_ext"].process_and_store(tb)
            table_names.append(tname)
        img_count = 0
        if result.images:
            img_count = components["orchestrator"].image_store.add_images(result.images)
        if text_count == 0 and not table_names and img_count == 0:
            # 解析成功但无任何内容，视为无效文件，回滚
            _delete_source(uf.name)
            return f"❌ {uf.name}: 未提取到任何有效内容（文件可能为空或格式异常）"
        parts = [f"{text_count}个文本块", f"{len(table_names)}张表"]
        if img_count:
            parts.append(f"{img_count}张截图")
        return f"{uf.name}: " + ", ".join(parts)
    except Exception as e:
        # 失败回滚：清除可能已部分写入的数据，避免脏数据
        try:
            _delete_source(uf.name)
        except Exception:
            pass
        logger.error(f"入库失败 {uf.name}: {e}")
        return f"❌ {uf.name}: 入库失败 - {e}"


# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    # ---- 知识库状态（所有人可见） ----
    st.subheader("📊 知识库状态")
    stats = components["vector_store"].get_stats()
    tables = components["table_ext"].get_all_tables_info()
    existing_files = _get_all_source_files()

    col_a, col_b = st.columns(2)
    col_a.metric("文本块", stats["total_chunks"])
    col_b.metric("数据表", len(tables))

    if existing_files:
        with st.expander(f"已入库文件 ({len(existing_files)})"):
            for f in existing_files:
                st.text(f"📄 {f}")

    st.divider()

    # ---- 导出（所有人可用） ----
    st.subheader("📤 数据导出")
    export_fmt = st.selectbox("导出格式", ["xlsx", "csv", "pdf", "docx"])
    if st.button("导出上次查询结果"):
        # 取最近一条助手回答：有表格则导表格，纯文本(制度问答)则导文本
        last_msg = next(
            (m for m in reversed(st.session_state.chat_history)
             if m["role"] == "assistant"),
            None,
        )
        if last_msg is None:
            st.info("暂无可导出的内容")
        else:
            try:
                data = last_msg.get("data")
                if data is not None and not data.empty:
                    # 导出前同样脱敏，避免绕过页面展示的脱敏（导出口径与展示一致）
                    data = components["security"].mask_dataframe(data)
                    path = components["exporter"].export(data, fmt=export_fmt)
                else:
                    path = components["exporter"].export_text(
                        answer=last_msg.get("content", ""),
                        question=last_msg.get("question", ""),
                        sources=last_msg.get("sources", []),
                        snippets=last_msg.get("snippets", []),
                        fmt=export_fmt,
                    )
                with open(path, "rb") as f:
                    st.download_button(
                        f"⬇️ 下载 {path.name}", f.read(), file_name=path.name,
                    )
            except ValueError as e:
                st.warning(str(e))

    st.divider()

    # ---- 管理员区域（需登录） ----
    st.subheader("🔧 文档管理")

    import time as _time
    # 会话超时自动登出
    if (st.session_state.admin_logged_in
            and st.session_state.login_time
            and _time.time() - st.session_state.login_time > SESSION_TIMEOUT_SECONDS):
        st.session_state.admin_logged_in = False
        st.session_state.login_time = 0.0
        st.warning(f"⏱️ 登录已超过 {SESSION_TIMEOUT_SECONDS // 60} 分钟，已自动退出，请重新登录")

    # 默认密码告警：仍在用样例密码时提示尽快修改
    if ADMIN_PASSWORD == "finance2025":
        st.warning("⚠️ 仍在使用默认管理员密码，请在 `.env` 设置 `ADMIN_PASSWORD` 后重启")

    if not st.session_state.admin_logged_in:
        with st.expander("管理员登录"):
            # 服务端级锁定：按用户名计，换浏览器会话/无痕窗口也无法绕过
            guard = _login_guard()
            admin_user = st.text_input("用户名", key="admin_user")
            admin_pwd = st.text_input("密码", type="password", key="admin_pwd")
            lock_key = admin_user or ADMIN_USER
            now = _time.time()
            lock_until = guard["lock_until"].get(lock_key, 0.0)
            locked = now < lock_until
            if locked:
                wait = int(lock_until - now)
                st.error(f"🔒 登录失败次数过多，请 {wait} 秒后重试")
            if st.button("登录", key="admin_login_btn", disabled=locked):
                pwd_hash = hashlib.sha256(admin_pwd.encode()).hexdigest()
                if admin_user in ADMINS and hmac.compare_digest(ADMINS[admin_user], pwd_hash):
                    st.session_state.admin_logged_in = True
                    st.session_state.login_time = _time.time()
                    with guard["lock"]:
                        guard["fails"].pop(lock_key, None)
                        guard["lock_until"].pop(lock_key, None)
                    st.rerun()
                else:
                    with guard["lock"]:   # 读改写加锁，防并发丢增量削弱锁定
                        fails = guard["fails"].get(lock_key, 0) + 1
                        guard["fails"][lock_key] = fails
                        locked_now = fails >= LOGIN_MAX_FAILS
                        if locked_now:
                            guard["lock_until"][lock_key] = _time.time() + LOGIN_LOCK_SECONDS
                            guard["fails"][lock_key] = 0
                    remain = LOGIN_MAX_FAILS - fails
                    if locked_now:
                        st.error(f"🔒 失败次数过多，账号锁定 {LOGIN_LOCK_SECONDS // 60} 分钟")
                    else:
                        st.error(f"用户名或密码错误（还可尝试 {remain} 次）")
    else:
        st.caption("✅ 已登录管理员")
        if st.button("退出管理", use_container_width=True):
            st.session_state.admin_logged_in = False
            st.session_state.login_time = 0.0
            st.rerun()

        doc_mode = st.radio(
            "操作模式",
            ["📥 追加入库", "🔄 覆盖替换", "🗑️ 删除文件"],
            horizontal=True,
        )

        # ---- 入库结果横幅（rerun 后展示一次）----
        if st.session_state.ingest_result:
            info = st.session_state.ingest_result
            details = info["details"]
            ok = [d for d in details if not d.startswith("❌")]
            failed = [d for d in details if d.startswith("❌")]
            if ok:
                st.success(f"🎉 入库成功！{len(ok)} 个文件已进入知识库")
                for line in ok:
                    st.caption(f"　✅ {line}")
            if failed:
                st.error(f"⚠️ {len(failed)} 个文件入库失败（已跳过，不影响其他文件）")
                for line in failed:
                    st.caption(f"　{line}")
            st.session_state.ingest_result = None

        # ---- 追加入库 ----
        if doc_mode == "📥 追加入库":
            st.caption("上传新文件，与现有知识库共存")
            uploaded_files = st.file_uploader(
                "选择文件",
                type=["pdf", "docx", "xlsx", "xls", "csv", "xlsm"],
                accept_multiple_files=True,
                key=f"upload_append_{st.session_state.uploader_seq}",
            )
            if uploaded_files:
                cache_warns = _check_excel_formula_cache(uploaded_files)
                if cache_warns:
                    names = "、".join(w["name"] for w in cache_warns)
                    st.warning(
                        f"⚠️ 检测到 **{names}** 含大量公式且可能未重算，"
                        "数值可能不准确。\n\n"
                        "👉 建议：先在 Excel 中打开该文件（Mac 会自动重算）→ Cmd+S 保存 → 再上传。\n\n"
                        "如已手动刷新过，可直接点击入库。"
                    )
                name_issues = _check_filename_quality(uploaded_files)
                if name_issues:
                    for item in name_issues:
                        tip = "、".join(item["problems"])
                        st.warning(f"⚠️ **{item['name']}** 文件名{tip}，可能影响系统识别公司归属和年份判断，建议修正后再上传。")
                if st.button("确认入库", type="primary"):
                    progress = st.progress(0)
                    details = []
                    for i, uf in enumerate(uploaded_files):
                        with st.spinner(f"解析: {uf.name}"):
                            msg = _ingest_file(uf)
                            details.append(msg)
                        progress.progress((i + 1) / len(uploaded_files))
                    st.session_state.ingest_result = {
                        "count": len(uploaded_files),
                        "details": details,
                    }
                    st.session_state.uploader_seq += 1
                    st.toast("🎉 入库成功！", icon="✅")
                    st.rerun()

        # ---- 覆盖替换 ----
        elif doc_mode == "🔄 覆盖替换":
            st.caption("选中要被替换的旧文件 → 上传新文件 → 删旧入新")
            if not existing_files:
                st.info("知识库为空，无可覆盖文件")
            else:
                targets = st.multiselect(
                    "选择要被替换的旧文件",
                    existing_files,
                    key="replace_targets",
                )
                uploaded_files = st.file_uploader(
                    "上传新文件（替换上面选中的旧文件）",
                    type=["pdf", "docx", "xlsx", "xls", "csv", "xlsm"],
                    accept_multiple_files=True,
                    key=f"upload_replace_{st.session_state.uploader_seq}",
                )
                if targets and uploaded_files:
                    cache_warns = _check_excel_formula_cache(uploaded_files)
                    if cache_warns:
                        names = "、".join(w["name"] for w in cache_warns)
                        st.warning(
                            f"⚠️ 检测到 **{names}** 含大量公式且可能未重算，"
                            "数值可能不准确。\n\n"
                            "👉 建议：先在 Excel 中打开该文件（Mac 会自动重算）→ Cmd+S 保存 → 再上传。\n\n"
                            "如已手动刷新过，可直接点击覆盖。"
                        )
                    name_issues = _check_filename_quality(uploaded_files)
                    if name_issues:
                        for item in name_issues:
                            tip = "、".join(item["problems"])
                            st.warning(f"⚠️ **{item['name']}** 文件名{tip}，可能影响系统识别公司归属和年份判断，建议修正后再上传。")
                if targets and uploaded_files and st.button("确认覆盖", type="primary"):
                    progress = st.progress(0)
                    total_steps = len(targets) + len(uploaded_files)
                    step = 0
                    details = []
                    for fname in targets:
                        with st.spinner(f"删除旧文件: {fname}"):
                            _delete_source(fname)
                        step += 1
                        progress.progress(step / total_steps)
                    for uf in uploaded_files:
                        with st.spinner(f"入库新文件: {uf.name}"):
                            msg = _ingest_file(uf)
                            details.append(msg)
                        step += 1
                        progress.progress(step / total_steps)
                    st.session_state.ingest_result = {
                        "count": len(uploaded_files),
                        "details": [f"已删除 {len(targets)} 个旧文件"] + details,
                    }
                    st.session_state.uploader_seq += 1
                    st.toast("🎉 覆盖入库成功！", icon="✅")
                    st.rerun()

        # ---- 删除文件 ----
        elif doc_mode == "🗑️ 删除文件":
            st.caption("选中文件从知识库中彻底移除")
            if not existing_files:
                st.info("知识库为空")
            else:
                targets = st.multiselect(
                    "选择要删除的文件",
                    existing_files,
                    key="delete_targets",
                )
                if targets:
                    st.warning(f"将删除 {len(targets)} 个文件的全部数据（不可恢复）")
                    if st.button("确认删除", type="primary"):
                        progress = st.progress(0)
                        for i, fname in enumerate(targets):
                            with st.spinner(f"删除: {fname}"):
                                _delete_source(fname)
                            progress.progress((i + 1) / len(targets))
                        st.success(f"已删除 {len(targets)} 个文件")
                        st.rerun()

# ============================================================
# 主区域：问答（无需登录）
# ============================================================
_t1, _t2 = st.columns([5, 1])
with _t1:
    st.title("💬 财务知识库问答系统")
with _t2:
    if st.session_state.chat_history:
        if st.button("🗑️ 清空对话", use_container_width=True,
                     help="清除当前对话历史，重新开始（不影响知识库）"):
            st.session_state.chat_history = []
            st.session_state.last_data = None
            st.rerun()
st.caption("支持财务数据查询（自动生成SQL）和规章制度查询（RAG检索），结果可导出Excel/CSV")
st.info("⚠️ 回答由 AI 从报表/文档自动生成，可能存在选表或提取偏差。**涉及决策请以每条回答下方「数据来源」标注的原始报表为准。**", icon="⚠️")

def _safe_upload_path(file: str) -> Path:
    """解析上传目录内的文件路径，阻断 ../ 路径遍历越权读取"""
    base = UPLOAD_DIR.resolve()
    p = (UPLOAD_DIR / file).resolve()
    if base != p and base not in p.parents:
        raise ValueError("非法文件路径")
    return p


@st.cache_data(show_spinner=False)
def _preview_sheet(file: str, sheet: str):
    """读取原始Excel的指定sheet用于页面预览（保留原貌）"""
    try:
        path = _safe_upload_path(file)
    except ValueError:
        return None
    if not path.exists():
        return None
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, header=None)
        return pd.read_excel(path, sheet_name=sheet, header=None)
    except Exception:
        return None


@st.fragment
def _preview_toggle_fragment(file: str, sheet: str, key: str):
    """局部 fragment：toggle 只触发自身 rerun，不引起整页重排（避免点击后页面向上滑）"""
    show_pv = st.toggle("预览命中表", key=key)
    if show_pv:
        df_prev = _preview_sheet(file, sheet)
        if df_prev is not None:
            st.dataframe(df_prev, use_container_width=True, height=300)
        else:
            st.warning("无法预览（原始文件可能已删除）")


@st.cache_data(show_spinner=False)
def _download_original(file: str):
    """读取原始上传文件的字节流，原封不动供下载"""
    path = _safe_upload_path(file)
    return path.read_bytes()


def _render_sources(sources, snippets=None, key_prefix="live", page_images=None):
    """渲染引用来源和原文片段"""
    # 数据来源置顶显眼展示：不折叠，让用户第一眼看到数字取自哪张表/哪个口径
    table_srcs = [s for s in (sources or []) if s.get("type") == "table_source"]
    if table_srcs:
        summary = "；".join(
            f"{s['file']}（{s.get('caliber','')}口径 · {s.get('sheet','')}）"
            for s in table_srcs
        )
        st.caption(f"📊 **本次数据来源**：{summary}　—　请与原始报表核对")

    if sources:
        with st.expander("📎 引用来源", expanded=False):
            for i, s in enumerate(sources):
                stype = s.get("type")

                # 数据查询来源：可预览/下载的Excel sheet
                if stype == "table_source":
                    caliber = s.get("caliber", "")
                    badge = f"［{caliber}口径］" if caliber else ""
                    st.markdown(f"📊 **{s['file']}** · 命中工作表：`{s.get('sheet','')}` {badge}")

                    # 预览（仅命中sheet）— 用 fragment 隔离 rerun 范围，避免整页重排导致滚动跳动
                    _preview_toggle_fragment(
                        s["file"], s.get("sheet", ""),
                        key=f"{key_prefix}_pv_{i}",
                    )
                    # 下载（完整原始文件，原封不动）
                    try:
                        fbytes = _download_original(s["file"])
                        st.download_button(
                            "⬇️ 下载原文件",
                            fbytes,
                            file_name=s["file"],
                            key=f"{key_prefix}_dl_{i}",
                        )
                    except Exception as e:
                        st.caption(f"下载不可用: {e}")
                    st.divider()

                # 文本RAG来源
                elif "file" in s:
                    loc = f"📄 **{s['file']}**"
                    if s.get("page"):
                        loc += f" · 第{s['page']}页"
                    if s.get("heading_path"):
                        loc += f" · {s['heading_path']}"
                    elif s.get("section"):
                        loc += f" · {s['section']}"
                    if s.get("relevance") is not None:
                        loc += f" · 相关度 {s.get('relevance', '-')}"
                    st.markdown(loc)
                    # 下载原文件按钮
                    try:
                        fbytes = _download_original(s["file"])
                        st.download_button(
                            "⬇️ 下载原文件",
                            fbytes,
                            file_name=s["file"],
                            key=f"{key_prefix}_textdl_{i}",
                        )
                    except Exception as e:
                        st.caption(f"下载不可用: {e}")
                    st.divider()

    if snippets:
        with st.expander("📄 原文片段", expanded=False):
            for i, sn in enumerate(snippets, 1):
                header = f"**片段{i}** — {sn['file']}"
                if sn.get("page"):
                    header += f", 第{sn['page']}页"
                if sn.get("heading_path"):
                    header += f", {sn['heading_path']}"
                st.markdown(header)
                st.info(sn["text"])

    # PDF 截图按需查看
    if page_images:
        st.caption(f"📷 相关页面包含 {len(page_images)} 张截图")
        with st.expander(f"查看截图（{len(page_images)}张）", expanded=False):
            for i, img in enumerate(page_images):
                p = Path(img["image_path"])
                if not p.exists():
                    continue
                caption = f"{img['source_file']} · 第{img['page_num']}页"
                if img.get("width") and img.get("height"):
                    caption += f" · {img['width']}×{img['height']}"
                try:
                    st.image(str(p), caption=caption, use_container_width=True)
                except Exception as e:
                    st.caption(f"图片加载失败: {e}")


def _render_feedback(idx: int, question: str, answer: str):
    """回答下方的 👍👎 反馈入口，写入 logs/feedback_log.jsonl 供评估用"""
    given = st.session_state.feedback.get(idx)
    if given:
        st.caption("已收到反馈 " + ("👍" if given == "up" else "👎") + "，感谢！")
        return
    c1, c2, _ = st.columns([1, 1, 10])
    if c1.button("👍", key=f"fb_up_{idx}", help="回答准确有用"):
        components["security"].log_feedback(question, answer, "up", _current_user_id())
        st.session_state.feedback[idx] = "up"
        st.rerun()
    if c2.button("👎", key=f"fb_down_{idx}", help="回答有误或没帮助"):
        components["security"].log_feedback(question, answer, "down", _current_user_id())
        st.session_state.feedback[idx] = "down"
        st.rerun()


for msg_idx, msg in enumerate(st.session_state.chat_history):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("data") is not None:
            _df = components["security"].mask_dataframe(msg["data"])
            with st.expander(f"📊 查询结果（{len(_df)} 行 × {len(_df.columns)} 列）",
                              expanded=False):
                st.dataframe(_df, use_container_width=True)
            chart_helper.render_chart_section(
                msg["data"], msg.get("question", msg.get("content", "")),
                key_prefix=f"chartmsg{msg_idx}",
            )
        _render_sources(msg.get("sources"), msg.get("snippets"),
                        key_prefix=f"hist{msg_idx}",
                        page_images=msg.get("page_images"))
        # 只对真实问答（带question字段）显示反馈入口，跳过警告类消息
        if msg["role"] == "assistant" and msg.get("question"):
            _render_feedback(msg_idx, msg["question"], msg["content"])

# 公司名同音纠错：待确认卡片（精确匹配落空且找到同音公司时显示）
if st.session_state.pending_confirm:
    pc = st.session_state.pending_confirm
    fixes = pc["fixes"]
    with st.chat_message("assistant"):
        if len(fixes) == 1:
            f = fixes[0]
            st.warning(f"⚠️ 知识库中没有「{f['wrong']}」这家公司。您是否指：**{f['name']}**？")
        else:
            lines = "\n".join(
                f"- 「**{f['wrong']}**」→ **{f['name']}**（简称：{f['right']}）"
                for f in fixes
            )
            st.warning(f"⚠️ 检测到 {len(fixes)} 处可能的公司名错别字：\n\n{lines}")

        right_summary = "、".join(f["right"] for f in fixes)
        cc1, cc2 = st.columns(2)
        if cc1.button(f"✅ 全部按「{right_summary}」查询", key="confirm_company_fix"):
            fixed = pc["original"]
            for f in fixes:
                fixed = fixed.replace(f["wrong"], f["right"])
            st.session_state._pending_run = fixed
            st.session_state.pending_confirm = None
            st.rerun()
        if cc2.button("❌ 不是，仍按我输入的查", key="reject_company_fix"):
            st.session_state._pending_run = pc["original"]
            st.session_state.pending_confirm = None
            st.rerun()

# 确定本轮要处理的问题：来自确认注入 或 来自输入框
_typed = st.chat_input("请输入问题，例如：2024年差旅费合计 / 报销流程是什么")
question = None
_skip_user_append = False
if st.session_state.get("_pending_run"):
    question = st.session_state.pop("_pending_run")
    _skip_user_append = True  # 用户原始输入已在历史中
elif _typed:
    # 先做公司名同音纠错检测
    st.session_state.pending_confirm = None
    fixes = company_matcher.detect_company_typo(_typed)
    if fixes:
        st.session_state.chat_history.append({"role": "user", "content": _typed})
        st.session_state.pending_confirm = {"fixes": fixes, "original": _typed}
        st.rerun()
    else:
        question = _typed

if question:
    if not _skip_user_append:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

    safety = components["security"].check_question_safety(question)
    if not safety["safe"]:
        with st.chat_message("assistant"):
            st.warning(safety["warning"])
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"⚠️ {safety['warning']}",
        })
    else:
        # 分阶段进度展示 + 流式输出
        STAGE_LABELS = {
            "context": "1/5 正在分析上下文",
            "route": "2/5 正在识别查询意图",
            "select_table": "3/5 正在匹配财务表",
            "generate_sql": "4/5 正在生成 SQL",
            "execute_sql": "4/5 正在执行查询",
            "summarize": "5/5 正在生成回答",
        }
        STAGE_PCT = {
            "context": 10, "route": 20, "select_table": 40,
            "generate_sql": 55, "execute_sql": 70, "summarize": 85,
        }

        with st.chat_message("assistant"):
            import time as _t
            start_ts = _t.time()
            status_box = st.status("正在思考…", expanded=True)
            stage_log = []

            def _on_step(stage, msg):
                label = STAGE_LABELS.get(stage, msg)
                pct = STAGE_PCT.get(stage, 50)
                stage_log.append(f"✓ {label}")
                elapsed = int(_t.time() - start_ts)
                hint = f"（已用 {elapsed}s）"
                if elapsed > 30:
                    hint += "  ⏳ AI推理较慢，请耐心等待…"
                status_box.update(label=f"{label}  {hint}", state="running")
                with status_box:
                    st.progress(pct, text=" / ".join(stage_log[-3:]))

            try:
                result = components["orchestrator"].ask(
                    question,
                    chat_history=st.session_state.chat_history[:-1],
                    on_step=_on_step,
                    stream=True,
                )
            except Exception as e:
                logger.error(f"查询失败: {question[:50]} → {e}")
                components["security"].log_query(
                    question, "error", str(e), [], success=False,
                    latency_ms=int((_t.time() - start_ts) * 1000),
                    user_id=_current_user_id(),
                )
                status_box.update(label="❌ 查询出错", state="error")
                friendly = (
                    "抱歉，处理这个问题时出现了异常，请稍后重试或换个问法。\n\n"
                    "常见原因：① AI 服务暂时不可用或超时；② 网络波动。"
                )
                st.error(friendly)
                st.session_state.chat_history.append({
                    "role": "assistant", "content": f"⚠️ {friendly}",
                })
                st.stop()

            # 数据查询：SQL已执行完，先把表格秒级展示给用户（脱敏后展示）
            if result.get("data") is not None and not result["data"].empty:
                data_ready_ts = int(_t.time() - start_ts)
                status_box.update(
                    label=f"数据已就绪（耗时 {data_ready_ts}s），AI正在生成解读…",
                    state="running",
                )
                masked_df = components["security"].mask_dataframe(result["data"])
                with st.expander(
                    f"📊 查询结果（{len(masked_df)} 行 × {len(masked_df.columns)} 列）",
                    expanded=True,
                ):
                    st.dataframe(masked_df, use_container_width=True)
                st.session_state.last_data = result["data"]

            answer_value = result["answer"]
            answer_box = st.empty()
            # 处理流式 generator
            if hasattr(answer_value, "__iter__") and not isinstance(answer_value, str):
                with answer_box.container():
                    full_answer = st.write_stream(answer_value)
            else:
                full_answer = answer_value

            # 完成后用脱敏版覆盖显示（流式过程中无法跨chunk匹配，故结束后统一脱敏重绘）
            full_answer = components["security"].mask_sensitive(full_answer)
            answer_box.markdown(full_answer)

            # 真正完成时才标记 complete
            elapsed = int(_t.time() - start_ts)
            status_box.update(label=f"✅ 完成（总耗时 {elapsed}s）", state="complete", expanded=False)

            # 渲染顺序必须与历史循环一致（图表在前、引用来源在后），
            # 否则点击图表触发 rerun 切到历史渲染时，图表会"跳"到引用来源上方
            # 财务数据问答：智能图表入口（适合可视化时才显示）
            if result.get("data") is not None and not result["data"].empty:
                chart_helper.render_chart_section(
                    result["data"], question,
                    key_prefix=f"chartmsg{len(st.session_state.chat_history)}",
                )

            # key_prefix 用 hist{N}：与历史循环 (msg_idx=N) 保持一致，
            # 否则下次 rerun toggle widget key 漂移、状态丢失，造成"首次点击不展开"
            _render_sources(result.get("sources"), result.get("snippets"),
                            key_prefix=f"hist{len(st.session_state.chat_history)}",
                            page_images=result.get("page_images"))

            _success = not (isinstance(full_answer, str)
                            and full_answer.startswith(("数据查询失败", "⚠️")))
            components["security"].log_query(
                question, result["type"], full_answer, result.get("sources", []),
                success=_success, sql=result.get("sql", ""),
                latency_ms=int((_t.time() - start_ts) * 1000),
                user_id=_current_user_id(),
            )
            _render_feedback(len(st.session_state.chat_history),
                             question, full_answer)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": full_answer,
            "question": question,
            "data": result.get("data"),
            "sources": result.get("sources", []),
            "snippets": result.get("snippets", []),
            "page_images": result.get("page_images", []),
        })

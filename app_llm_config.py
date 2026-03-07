"""项目统一配置文件。

说明：
1) `app.py` 读取本文件中的 UI 与流程配置。
2) `llm.py` 读取本文件中的模型与安全策略配置。
3) `deal.py` 读取本文件中的 ALNS 默认参数。
"""

import os
from dataclasses import dataclass


def _read_api_key() -> str:
    """读取模型 API Key。

    优先级：
    1) 本地环境变量 `DEEPSEEK_API_KEY`
    2) Streamlit secrets 顶层 `DEEPSEEK_API_KEY`
    3) Streamlit secrets 分组 `llm.api_key`
    """
    env_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if env_key:
        return env_key

    try:
        import streamlit as st
    except Exception:
        return ""

    try:
        secrets = st.secrets
    except Exception:
        return ""

    top_key = str(secrets.get("DEEPSEEK_API_KEY", "")).strip()
    if top_key:
        return top_key

    llm_group = secrets.get("llm")
    if hasattr(llm_group, "get"):
        group_key = str(llm_group.get("api_key", "")).strip()
        if group_key:
            return group_key

    return ""


# ------------------------
# app.py 配置
# ------------------------
@dataclass(frozen=True)
class AppUIConfig:
    """Streamlit 页面基础配置。"""

    page_title: str = "排产前端工作台"
    page_icon: str = "📊"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"


APP_UI_CONFIG = AppUIConfig()

# 会话目录与 LLM 模块名（动态导入使用）。
APP_SESSIONS_DIR = "sessions"
APP_LLM_MODULE = "llm"

# 开发期可启用 llm 模块自动重载（默认关闭，避免清空 llm 会话内存）。
APP_LLM_AUTO_RELOAD = os.getenv("APP_LLM_AUTO_RELOAD", "0").strip() == "1"

# app.py 调用 llm.py 的函数名约定。
APP_LLM_IMPORT_PARAM_FN = "import_param_json"
APP_LLM_FEEDBACK_FN = "handle_user_feedback"
APP_LLM_IMPORT_RESULT_FN = "import_final_result"
APP_LLM_DEBUG_FN = "get_session_debug_state"
APP_LLM_PREVIEW_ALGO_FN = "preview_algorithm_payload"

# 前端流程步骤顺序（用于进度条和状态机）。
APP_STEP_ORDER = [
    "draft",
    "awaiting_llm_json",
    "awaiting_confirmation",
    "awaiting_final_result",
    "completed",
]

# 前端流程步骤中文文案。
APP_STEP_TEXT = {
    "draft": "1) 输入调整约束",
    "awaiting_llm_json": "2) 生成约束参数 JSON（可重试）",
    "awaiting_confirmation": "3) 用户确认约束参数",
    "awaiting_final_result": "4) 框架执行中（自动回传结果）",
    "completed": "5) 展示完成",
}


# ------------------------
# llm.py 配置
# ------------------------
@dataclass(frozen=True)
class LLMRuntimeConfig:
    """LLM 运行时参数。"""

    model_name: str = os.getenv("LLM_MODEL", "deepseek-chat")
    api_key: str = _read_api_key()
    api_base: str = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))


LLM_RUNTIME_CONFIG = LLMRuntimeConfig()

# llm.py 会话缓存上限，超过后会淘汰最早会话。
LLM_SESSION_CACHE_MAX = int(os.getenv("LLM_SESSION_CACHE_MAX", "200"))

# 安全校验中用于检测可疑注入片段的正则模式。
SAFETY_SUSPICIOUS_PATTERNS = [
    r"<script",
    r"drop\s+table",
    r"union\s+select",
    r"--",
]

# 禁止出现的敏感字段名（统一按小写比较）。
SAFETY_BLOCKED_KEYS = {
    "password",
    "passwd",
    "api_key",
    "apikey",
    "token",
    "secret",
    "private_key",
}

# 安全校验里要求至少出现其一的业务关键字段。
SAFETY_REQUIRED_ANY_KEYS = {
    "jobs",
    "machines",
    "constraints",
    "objective",
    "task_type",
    "algorithm_parameters",
}


# ------------------------
# deal.py ALNS 配置
# ------------------------
@dataclass(frozen=True)
class DealALNSConfig:
    """ALNS 求解默认参数。"""

    instance_path: str = "la05.json"
    rng_seed: int = 42
    alns_iterations: int = 250
    destroy_ratio: float = 0.2
    initial_temperature: float = 25.0
    cooling_rate: float = 0.992
    min_temperature: float = 0.2
    weight_alpha: float = 0.2
    failure_block_horizon: int = 10_000


DEAL_ALNS_CONFIG = DealALNSConfig()

# deal.py 结果缓存上限，超过后会淘汰最早结果。
DEAL_RESULT_CACHE_MAX = int(os.getenv("DEAL_RESULT_CACHE_MAX", "200"))

import os
from dataclasses import dataclass


def _read_api_key() -> str:
    """优先读取本地环境变量，读取不到再读取 Streamlit secrets。"""
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
    page_title: str = "排产前端工作台"
    page_icon: str = "🧭"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"


APP_UI_CONFIG = AppUIConfig()

APP_SESSIONS_DIR = "sessions"
APP_LLM_MODULE = "llm"

APP_LLM_IMPORT_PARAM_FN = "import_param_json"
APP_LLM_FEEDBACK_FN = "handle_user_feedback"
APP_LLM_IMPORT_RESULT_FN = "import_final_result"
APP_LLM_DEBUG_FN = "get_session_debug_state"

APP_STEP_ORDER = [
    "draft",
    "awaiting_llm_json",
    "awaiting_confirmation",
    "awaiting_final_result",
    "completed",
]

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
    model_name: str = os.getenv("LLM_MODEL", "deepseek-chat")
    api_key: str = _read_api_key()
    api_base: str = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))


LLM_RUNTIME_CONFIG = LLMRuntimeConfig()

SAFETY_SUSPICIOUS_PATTERNS = [
    r"<script",
    r"drop\s+table",
    r"union\s+select",
    r"--",
]

SAFETY_BLOCKED_KEYS = {
    "password",
    "passwd",
    "api_key",
    "apikey",
    "token",
    "secret",
    "private_key",
}

SAFETY_REQUIRED_ANY_KEYS = {
    "jobs",
    "machines",
    "constraints",
    "objective",
    "task_type",
    "requirement",
}


# ------------------------
# deal.py ALNS 配置
# ------------------------
@dataclass(frozen=True)
class DealALNSConfig:
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



DEAL_ALNS_CONFIG = DealALNSConfig()



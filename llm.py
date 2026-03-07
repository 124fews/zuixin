from __future__ import annotations

"""LLM 编排层。

职责：
1) 将自然语言约束转成结构化 JSON 参数；
2) 做参数安全校验；
3) 与算法执行层（deal.py）对接；
4) 维护会话级记忆（反馈历史、最近参数、调试信息）。
"""

import importlib
import hashlib
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from app_llm_config import (
    LLM_RUNTIME_CONFIG,
    LLM_SESSION_CACHE_MAX,
    SAFETY_BLOCKED_KEYS,
    SAFETY_REQUIRED_ANY_KEYS,
    SAFETY_SUSPICIOUS_PATTERNS,
)

LANGCHAIN_AVAILABLE = True
LANGCHAIN_IMPORT_ERROR = ""

try:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
except Exception as error:  # pragma: no cover
    LANGCHAIN_AVAILABLE = False
    LANGCHAIN_IMPORT_ERROR = str(error)


SESSION_MEMORY: Dict[str, Dict[str, Any]] = {}

FAILURE_HINT_KEYWORDS = [
    "故障",
    "错误",
    "停机",
    "不可用",
    "不能用",
    "不可使用",
    "停用",
    "宕机",
    "维护",
    "down",
    "failure",
    "unavailable",
    "out of service",
]


def _evict_old_sessions_if_needed(max_size: int = LLM_SESSION_CACHE_MAX) -> None:
    """限制会话缓存大小，避免长期运行内存持续增长。"""
    if max_size <= 0:
        return
    while len(SESSION_MEMORY) >= max_size:
        oldest_key = next(iter(SESSION_MEMORY))
        SESSION_MEMORY.pop(oldest_key, None)


def _compute_payload_digest(payload: Dict[str, Any]) -> str:
    """计算参数 JSON 的稳定摘要，用于结果一致性校验。"""
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def _get_session_state(session_id: str) -> Dict[str, Any]:
    """获取会话状态；不存在则按默认结构初始化。"""

    if session_id not in SESSION_MEMORY:
        _evict_old_sessions_if_needed()
        SESSION_MEMORY[session_id] = {
            "feedback_history": [],
            "last_params": None,
            "last_semantics": None,
            "last_safety_report": None,
            "last_rag_context": [],
            "algorithm_dispatch": None,
            "last_submitted_payload_digest": None,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
    return SESSION_MEMORY[session_id]


def _touch_session(session_id: str) -> None:
    """更新会话的最后活跃时间戳。"""

    _get_session_state(session_id)["updated_at"] = datetime.now().isoformat(timespec="seconds")


def _extract_json_candidates(raw_text: str) -> List[str]:
    """从原始文本中提取可能的 JSON 片段候选。"""
    text = raw_text.strip()
    if not text:
        return []

    candidates: List[str] = [text]
    lines = text.splitlines()

    if len(lines) >= 3 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        candidates.append("\n".join(lines[1:-1]).strip())

    if "{" in text and "}" in text:
        candidates.append(text[text.find("{") : text.rfind("}") + 1].strip())
    if "[" in text and "]" in text:
        candidates.append(text[text.find("[") : text.rfind("]") + 1].strip())

    unique: List[str] = []
    seen = set()
    for item in candidates:
        if item and item not in seen:
            unique.append(item)
            seen.add(item)
    return unique


def _parse_json_flexible(value: Any) -> Tuple[Optional[Any], Optional[str]]:
    """尽量把输入解析为 JSON（兼容 dict/list/字符串包裹格式）。"""

    if isinstance(value, (dict, list)):
        return value, None

    if not isinstance(value, str):
        return None, f"类型不支持: {type(value).__name__}"

    for candidate in _extract_json_candidates(value):
        try:
            return json.loads(candidate), None
        except json.JSONDecodeError:
            pass

    return None, "无法从文本中解析合法 JSON"


def _build_llm() -> ChatOpenAI:
    """构建 LangChain ChatOpenAI 客户端。"""

    if not LANGCHAIN_AVAILABLE:
        raise RuntimeError(f"LangChain 依赖不可用: {LANGCHAIN_IMPORT_ERROR}")

    if not LLM_RUNTIME_CONFIG.api_key:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY，请在环境变量或 Streamlit secrets 中配置")

    return ChatOpenAI(
        model=LLM_RUNTIME_CONFIG.model_name,
        openai_api_key=LLM_RUNTIME_CONFIG.api_key,
        openai_api_base=LLM_RUNTIME_CONFIG.api_base,
        temperature=LLM_RUNTIME_CONFIG.temperature,
        streaming=False,
    )


def _llm_invoke_json(system_prompt: str, human_prompt: str, variables: Dict[str, Any]) -> Tuple[Optional[Any], Optional[str]]:
    """调用 LLM 并将返回内容解析为 JSON。"""

    try:
        llm = _build_llm()
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", human_prompt),
            ]
        )
        chain = prompt | llm | StrOutputParser()
        raw = chain.invoke(variables)
    except Exception as error:
        return None, str(error)

    return _parse_json_flexible(raw)


def _fallback_semantic_extract(requirement: str) -> Dict[str, Any]:
    """语义提取兜底逻辑：无法调用大模型时的规则降级。"""

    numbers = [int(num) for num in re.findall(r"\d+", requirement)]
    return {
        "intent": "production_rescheduling",
        "summary": requirement,
        "constraints": [],
        "priority": "normal",
        "numbers_mentioned": numbers,
        "entities": [],
        "assumptions": ["该结果来自规则降级流程，未调用大模型"],
    }


def _extract_machine_for_la05(text: str) -> List[str]:
    """从文本中抽取机器标识，统一输出为 `machine_{n}`。"""

    if not isinstance(text, str):
        return []

    machines: List[str] = []
    patterns = [
        r"(?:机器|机)\s*(\d+)\s*(?:号)?",
        r"\bmachine[_\s-]?(\d+)\b",
        r"\bM\s*(\d+)\b",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            machine_num = int(match)
            # 统一按 1-based 文本表达保留，deal.py 会做归一化。
            machines.append(f"machine_{machine_num}")
    return list(dict.fromkeys(machines))


def _has_failure_hint(text: str) -> bool:
    """判断文本是否包含故障/不可用语义线索。"""

    lowered = (text or "").lower()
    return any(keyword in lowered for keyword in FAILURE_HINT_KEYWORDS)


def _fallback_param_json(
    requirement: str,
    semantics: Dict[str, Any],
    rag_context: List[Dict[str, Any]],
    feedback_history: List[str],
) -> Dict[str, Any]:
    """参数生成兜底逻辑：按规则组装算法参数 JSON。"""

    semantic_summary = str(semantics.get("summary", ""))
    semantic_constraints = " ".join(str(item) for item in semantics.get("constraints", []))
    semantic_entities = " ".join(str(item) for item in semantics.get("entities", []))
    text_blob = f"{requirement} {semantic_summary} {semantic_constraints} {semantic_entities}"

    failure_hint = _has_failure_hint(text_blob)
    affected_machines = _extract_machine_for_la05(text_blob)

    algorithm_parameters: Dict[str, Any] = {
        "affected_machines": affected_machines,
        "failure_status": bool(failure_hint),
        "recovery_required": bool(failure_hint),
        "constraints_list": ["machine_unavailable"] if failure_hint else [],
        "action": "schedule_maintenance" if failure_hint else "reschedule",
    }

    return {
        "task_type": "reschedule",
        "requirement": requirement,
        "semantic": semantics,
        "constraints": semantics.get("constraints", []),
        "feedback_history": feedback_history,
        "rag_context": rag_context,
        "objective": "minimize_makespan",
        "algorithm_parameters": algorithm_parameters,
        "notes": ["fallback_mode"],
    }


def _enrich_algorithm_parameters(params_json: Dict[str, Any]) -> Dict[str, Any]:
    """补齐并归一化 algorithm_parameters 关键字段。"""

    params = dict(params_json)

    if "task_type" not in params:
        params["task_type"] = "reschedule"
    if "objective" not in params:
        params["objective"] = "minimize_makespan"

    if not isinstance(params.get("algorithm_parameters"), dict):
        params["algorithm_parameters"] = {}

    algo_params: Dict[str, Any] = params["algorithm_parameters"]
    requirement = str(params.get("requirement", ""))
    joined_constraints = " ".join(str(item) for item in params.get("constraints", []))
    semantic = params.get("semantic", {})
    semantic_summary = str(semantic.get("summary", "")) if isinstance(semantic, dict) else ""
    semantic_constraints = " ".join(str(item) for item in semantic.get("constraints", [])) if isinstance(semantic, dict) else ""
    semantic_entities = " ".join(str(item) for item in semantic.get("entities", [])) if isinstance(semantic, dict) else ""
    algo_constraints_list = " ".join(str(item) for item in algo_params.get("constraints_list", []))
    action_text = str(algo_params.get("action", "")).lower()
    text_blob = " ".join(
        [
            requirement,
            joined_constraints,
            semantic_summary,
            semantic_constraints,
            semantic_entities,
            algo_constraints_list,
            action_text,
        ]
    )

    failure_hint = _has_failure_hint(text_blob)
    if "maintenance" in action_text or "unavailable" in action_text:
        failure_hint = True

    extracted_from_text = _extract_machine_for_la05(text_blob)
    existing_machines_raw = algo_params.get("affected_machines", [])
    existing_machines: List[str] = []
    if isinstance(existing_machines_raw, (list, tuple, set)):
        for item in existing_machines_raw:
            existing_machines.extend(_extract_machine_for_la05(str(item)))
    elif existing_machines_raw is not None:
        existing_machines.extend(_extract_machine_for_la05(str(existing_machines_raw)))

    merged_machines = list(dict.fromkeys(existing_machines + extracted_from_text))
    if merged_machines:
        algo_params["affected_machines"] = merged_machines

    if failure_hint:
        algo_params.setdefault("failure_status", True)
        algo_params.setdefault("recovery_required", True)
        algo_params.setdefault("action", "schedule_maintenance")

    return params


def semantic_extract(
    requirement: str,
    messages: Optional[List[Dict[str, Any]]] = None,
    feedback_history: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """调用 LLM 做语义提取；失败时回退到规则提取。"""

    feedback_history = feedback_history or []
    messages = messages or []

    system_prompt = (
        "你是工业排产需求理解助手。"
        "请把用户需求提取为严格 JSON，字段包含: "
        "intent, summary, entities, constraints, objective, priority, assumptions。"
        "只能输出 JSON，不要 markdown。"
    )
    human_prompt = (
        "需求: {requirement}\n"
        "历史消息(可为空): {messages_json}\n"
        "用户驳回意见历史(可为空): {feedback_json}\n"
        "输出严格 JSON。"
    )
    variables = {
        "requirement": requirement,
        "messages_json": json.dumps(messages, ensure_ascii=False),
        "feedback_json": json.dumps(feedback_history, ensure_ascii=False),
    }

    parsed, error = _llm_invoke_json(system_prompt, human_prompt, variables)
    if error or not isinstance(parsed, dict):
        fallback = _fallback_semantic_extract(requirement)
        return fallback, {
            "mode": "fallback",
            "reason": error or "语义提取结果不是 JSON 对象",
        }

    return parsed, {"mode": "langchain"}


# ------------------------
# RAG 接口（预留）
# ------------------------
def fetch_rag_context(requirement: str, semantic_json: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    RAG 检索接口占位：
    后续可在这里接入向量库检索，将结果写入 rag_context。
    """
    _ = (requirement, semantic_json, top_k)
    return []


def generate_param_json(
    requirement: str,
    semantic_json: Dict[str, Any],
    rag_context: List[Dict[str, Any]],
    feedback_history: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """调用 LLM 生成算法参数 JSON；失败时回退规则生成。"""

    feedback_history = feedback_history or []

    system_prompt = (
        "你是排产参数生成助手。"
        "请根据需求语义、RAG上下文、历史驳回意见，生成给算法端使用的严格 JSON 参数。"
        "禁止输出 markdown，仅输出 JSON。"
    )
    human_prompt = (
        "需求原文: {requirement}\n"
        "语义提取: {semantic_json}\n"
        "RAG上下文: {rag_json}\n"
        "驳回意见历史: {feedback_json}\n"
        "请输出严格 JSON。"
    )
    variables = {
        "requirement": requirement,
        "semantic_json": json.dumps(semantic_json, ensure_ascii=False),
        "rag_json": json.dumps(rag_context, ensure_ascii=False),
        "feedback_json": json.dumps(feedback_history, ensure_ascii=False),
    }

    parsed, error = _llm_invoke_json(system_prompt, human_prompt, variables)
    if error or not isinstance(parsed, dict):
        fallback = _fallback_param_json(requirement, semantic_json, rag_context, feedback_history)
        return fallback, {
            "mode": "fallback",
            "reason": error or "参数生成结果不是 JSON 对象",
        }

    return parsed, {"mode": "langchain"}


SUSPICIOUS_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) for pattern in SAFETY_SUSPICIOUS_PATTERNS
]


def _walk_json(value: Any, path: str = "$"):
    """递归遍历 JSON 值，产出 (路径, 值) 迭代器。"""

    yield path, value
    if isinstance(value, dict):
        for key, sub_value in value.items():
            yield from _walk_json(sub_value, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, sub_value in enumerate(value):
            yield from _walk_json(sub_value, f"{path}[{idx}]")


def validate_param_json_safety(params_json: Dict[str, Any]) -> Dict[str, Any]:
    """对参数 JSON 做基础安全检测并返回风险报告。"""

    issues: List[Dict[str, Any]] = []

    if not isinstance(params_json, dict):
        issues.append({"level": "high", "path": "$", "message": "参数必须为 JSON 对象(dict)"})
        return {
            "passed": False,
            "risk_level": "high",
            "issues": issues,
        }

    if not any(key in params_json for key in SAFETY_REQUIRED_ANY_KEYS):
        issues.append(
            {
                "level": "medium",
                "path": "$",
                "message": "缺少关键业务字段，建议至少包含 jobs/machines/constraints/objective/task_type/algorithm_parameters 中之一",
            }
        )

    for path, value in _walk_json(params_json):
        if isinstance(value, dict):
            for key in value.keys():
                if str(key).lower() in SAFETY_BLOCKED_KEYS:
                    issues.append({"level": "high", "path": f"{path}.{key}", "message": "出现敏感字段"})

        if isinstance(value, str):
            if len(value) > 4000:
                issues.append({"level": "medium", "path": path, "message": "字符串过长，存在注入/污染风险"})
            for pattern in SUSPICIOUS_PATTERNS:
                if pattern.search(value):
                    issues.append(
                        {
                            "level": "high",
                            "path": path,
                            "message": f"检测到可疑片段: {pattern.pattern}",
                        }
                    )

    high_count = sum(1 for item in issues if item["level"] == "high")
    medium_count = sum(1 for item in issues if item["level"] == "medium")

    if high_count > 0:
        risk_level = "high"
        passed = False
    elif medium_count > 0:
        risk_level = "medium"
        passed = True
    else:
        risk_level = "low"
        passed = True

    return {
        "passed": passed,
        "risk_level": risk_level,
        "issues": issues,
        "summary": {
            "high": high_count,
            "medium": medium_count,
            "total": len(issues),
        },
    }


def _load_deal_module():
    # 不做 reload，避免清空 deal.py 内存中的结果缓存。
    # 只做按名称导入，保持算法层缓存与会话结果稳定。
    return importlib.import_module("deal")


def _prepare_algorithm_payload(params_json: Dict[str, Any]) -> Dict[str, Any]:
    """调用 deal.prepare_algorithm_payload，生成算法生效参数视图。"""

    deal = _load_deal_module()
    if hasattr(deal, "prepare_algorithm_payload"):
        return deal.prepare_algorithm_payload(params_json)
    return params_json


def submit_to_algorithm(requirement: str, params_json: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """将参数提交给算法层执行。"""

    try:
        deal = _load_deal_module()
        effective_params = _prepare_algorithm_payload(params_json)
        payload_digest = _compute_payload_digest(effective_params)
        result = deal.submit_job(requirement=requirement, params_json=effective_params, session_id=session_id)
        if isinstance(result, dict):
            result["effective_params"] = effective_params
            result["params_digest"] = payload_digest

        state = _get_session_state(session_id)
        state["last_submitted_payload_digest"] = payload_digest
        _touch_session(session_id)
        return result
    except Exception as error:
        return {
            "accepted": False,
            "job_id": session_id,
            "message": f"调用 deal.submit_job 失败: {error}",
        }


def preview_algorithm_payload(
    requirement: str,
    params_json: Dict[str, Any],
    session_id: str = "default",
    messages: Optional[List[Dict[str, Any]]] = None,
    workflow_step: Optional[str] = None,
) -> Dict[str, Any]:
    """返回“即将注入算法”的有效参数（用于前端回显）。"""

    _ = (requirement, messages, workflow_step)
    if not isinstance(params_json, dict):
        parsed, error = _parse_json_flexible(params_json)
        if error or not isinstance(parsed, dict):
            return {"error": "params_json 不是合法 JSON 对象"}
        params_json = parsed

    try:
        effective_params = _prepare_algorithm_payload(params_json)
    except Exception as error:
        return {"error": f"预览算法注入参数失败: {error}"}

    state = _get_session_state(session_id)
    state["last_params"] = params_json
    state["last_effective_params"] = effective_params
    _touch_session(session_id)
    return {"algorithm_payload": effective_params}


def fetch_algorithm_result(requirement: str, params_json: Dict[str, Any], session_id: str) -> Optional[Any]:
    """从算法层按 session_id 拉取结果。"""

    _ = (requirement, params_json)
    try:
        deal = _load_deal_module()
        state = _get_session_state(session_id)
        expected_digest: Optional[str] = state.get("last_submitted_payload_digest")
        requested_digest: Optional[str] = None
        if isinstance(params_json, dict):
            try:
                requested_digest = _compute_payload_digest(params_json)
            except Exception:
                requested_digest = None

        # 当前请求参数与最近一次提交参数不一致，说明该参数尚未执行，不返回旧结果。
        if expected_digest and requested_digest and expected_digest != requested_digest:
            return None
        if not expected_digest:
            expected_digest = requested_digest

        get_result = getattr(deal, "get_job_result", None)
        if not callable(get_result):
            return None

        try:
            return get_result(session_id, expected_digest=expected_digest)
        except TypeError:
            # 兼容旧版 deal.get_job_result(session_id) 签名
            return get_result(session_id)
    except Exception:
        return None


def import_param_json(
    requirement: str,
    session_id: str,
    messages: Optional[List[Dict[str, Any]]] = None,
    workflow_step: Optional[str] = None,
) -> Dict[str, Any]:
    """主入口：需求 -> 语义 -> 参数 -> 安全校验 -> 返回前端。"""

    _ = workflow_step
    requirement = (requirement or "").strip()
    if not requirement:
        return {"error": "需求不能为空"}

    state = _get_session_state(session_id)
    feedback_history: List[str] = state.get("feedback_history", [])

    semantic_json, semantic_meta = semantic_extract(requirement, messages=messages, feedback_history=feedback_history)
    rag_context = fetch_rag_context(requirement=requirement, semantic_json=semantic_json, top_k=5)
    params_json, params_meta = generate_param_json(
        requirement=requirement,
        semantic_json=semantic_json,
        rag_context=rag_context,
        feedback_history=feedback_history,
    )

    if not isinstance(params_json, dict):
        return {"error": "参数生成失败：返回结果不是 JSON 对象"}
    params_json.setdefault("requirement", requirement)
    params_json = _enrich_algorithm_parameters(params_json)
    safety_report = validate_param_json_safety(params_json)

    state["last_semantics"] = semantic_json
    state["last_rag_context"] = rag_context
    state["last_params"] = params_json
    state["last_safety_report"] = safety_report
    _touch_session(session_id)

    if not safety_report["passed"]:
        issue_preview = "; ".join(item["message"] for item in safety_report["issues"][:3])
        return {
            "error": f"安全校验未通过: {issue_preview}",
            "safety_report": safety_report,
        }

    return {
        "params_json": params_json,
        "meta": {
            "semantic_mode": semantic_meta.get("mode"),
            "semantic_reason": semantic_meta.get("reason"),
            "params_mode": params_meta.get("mode"),
            "params_reason": params_meta.get("reason"),
            "rag_hits": len(rag_context),
            "safety": safety_report,
        },
    }


def handle_user_feedback(
    requirement: str,
    params_json: Dict[str, Any],
    is_correct: bool,
    user_feedback: Optional[str] = None,
    session_id: str = "default",
    messages: Optional[List[Dict[str, Any]]] = None,
    workflow_step: Optional[str] = None,
) -> Dict[str, Any]:
    """处理用户确认/驳回，并在确认后触发算法提交。"""

    _ = (messages, workflow_step)
    state = _get_session_state(session_id)

    if not isinstance(params_json, dict):
        parsed, error = _parse_json_flexible(params_json)
        if error or not isinstance(parsed, dict):
            return {"error": "params_json 不是合法 JSON 对象"}
        params_json = parsed

    state["last_params"] = params_json

    if is_correct:
        dispatch_result = submit_to_algorithm(requirement=requirement, params_json=params_json, session_id=session_id)
        state["algorithm_dispatch"] = dispatch_result
        _touch_session(session_id)
        if not isinstance(dispatch_result, dict) or not dispatch_result.get("accepted"):
            dispatch_message = ""
            if isinstance(dispatch_result, dict):
                dispatch_message = str(dispatch_result.get("message", "算法端未接受任务"))
            return {
                "error": f"参数确认后提交算法端失败: {dispatch_message or '未知错误'}",
                "dispatch_result": dispatch_result,
            }
        return {
            "status": "confirmed",
            "message": "参数已确认并尝试提交到算法端",
            "dispatch_result": dispatch_result,
        }

    feedback_text = (user_feedback or "").strip()
    if not feedback_text:
        return {"error": "参数驳回时必须提供 user_feedback"}

    state.setdefault("feedback_history", []).append(feedback_text)
    _touch_session(session_id)

    return {
        "status": "rejected",
        "message": "已记录驳回意见，下一次导入参数会自动参考该意见",
        "feedback_count": len(state.get("feedback_history", [])),
    }


def import_final_result(
    requirement: str,
    params_json: Dict[str, Any],
    session_id: str,
    messages: Optional[List[Dict[str, Any]]] = None,
    workflow_step: Optional[str] = None,
) -> Dict[str, Any]:
    """导入并解析最终算法结果，供前端展示。"""

    _ = (messages, workflow_step)
    result = fetch_algorithm_result(requirement=requirement, params_json=params_json, session_id=session_id)

    if result is None:
        return {"error": "算法结果尚不可用（尚未确认参数或求解失败）"}

    parsed, error = _parse_json_flexible(result)
    if error:
        return {"error": f"算法结果不是合法 JSON: {error}"}

    _touch_session(session_id)
    return {"final_result": parsed}


def get_session_debug_state(session_id: str) -> Dict[str, Any]:
    """返回会话调试状态快照。"""

    return _get_session_state(session_id)


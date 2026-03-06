import importlib
import inspect
import json
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import streamlit as st
from app_llm_config import (
    APP_LLM_DEBUG_FN,
    APP_LLM_FEEDBACK_FN,
    APP_LLM_IMPORT_PARAM_FN,
    APP_LLM_IMPORT_RESULT_FN,
    APP_LLM_MODULE,
    APP_SESSIONS_DIR,
    APP_STEP_ORDER,
    APP_STEP_TEXT,
    APP_UI_CONFIG,
)

"""
Streamlit 前端入口：
1) 用户输入调整约束
2) 自动调用 llm.py 生成参数 JSON
3) 用户确认/驳回（回传 llm.py 处理）
4) 参数确认后自动触发框架求解并回显最终结果
"""

# 必须尽量作为首个 Streamlit 命令执行；
# 某些运行方式下若已被上层脚本设置过，这里忽略重复设置错误。
try:
    st.set_page_config(
        page_title=APP_UI_CONFIG.page_title,
        page_icon=APP_UI_CONFIG.page_icon,
        layout=APP_UI_CONFIG.layout,
        initial_sidebar_state=APP_UI_CONFIG.initial_sidebar_state,
    )
except Exception as error:
    if "set_page_config" not in str(error):
        raise


# ------------------------
# 全局常量（前端流程编排）
# ------------------------
SESSIONS_DIR = APP_SESSIONS_DIR
LLM_MODULE = APP_LLM_MODULE
LLM_IMPORT_PARAM_FN = APP_LLM_IMPORT_PARAM_FN
LLM_FEEDBACK_FN = APP_LLM_FEEDBACK_FN
LLM_IMPORT_RESULT_FN = APP_LLM_IMPORT_RESULT_FN
LLM_DEBUG_FN = APP_LLM_DEBUG_FN
STEP_ORDER = APP_STEP_ORDER
STEP_TEXT = APP_STEP_TEXT


def generate_session_name() -> str:
    """生成默认会话名（时间戳），用于会话文件命名。"""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_session_dir() -> None:
    """确保会话存储目录存在。"""
    if not os.path.exists(SESSIONS_DIR):
        os.makedirs(SESSIONS_DIR, exist_ok=True)


def init_state() -> None:
    """初始化页面运行所需的 session_state 默认值。"""
    defaults = {
        "current_session": generate_session_name(),
        "messages": [],
        "workflow_step": "draft",
        "user_requirement": "",
        "llm_params": None,
        "llm_meta": None,
        "reject_feedback": "",
        "feedback_result": None,
        "final_result": None,
        "outbox_events": [],
        "llm_debug_state": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_workflow() -> None:
    """仅重置当前业务流程状态，不更换会话 ID。"""
    st.session_state.workflow_step = "draft"
    st.session_state.user_requirement = ""
    st.session_state.llm_params = None
    st.session_state.llm_meta = None
    st.session_state.reject_feedback = ""
    st.session_state.feedback_result = None
    st.session_state.final_result = None
    st.session_state.llm_debug_state = None


def get_session_file(session_name: str) -> str:
    """根据会话名拼接会话文件路径。"""
    return os.path.join(SESSIONS_DIR, f"{session_name}.json")


def save_session() -> None:
    """把当前页面状态落盘到 sessions/{session}.json。"""
    if not st.session_state.current_session:
        return

    ensure_session_dir()
    data = {
        "current_session": st.session_state.current_session,
        "messages": st.session_state.messages,
        "workflow_step": st.session_state.workflow_step,
        "user_requirement": st.session_state.user_requirement,
        "llm_params": st.session_state.llm_params,
        "llm_meta": st.session_state.llm_meta,
        "reject_feedback": st.session_state.reject_feedback,
        "feedback_result": st.session_state.feedback_result,
        "final_result": st.session_state.final_result,
        "outbox_events": st.session_state.outbox_events,
    }
    with open(get_session_file(st.session_state.current_session), "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_sessions() -> List[str]:
    """加载本地会话列表并按时间倒序排列。"""
    ensure_session_dir()
    sessions: List[str] = []
    for filename in os.listdir(SESSIONS_DIR):
        if filename.endswith(".json"):
            sessions.append(filename[:-5])
    sessions.sort(reverse=True)
    return sessions


def load_session(session_name: str) -> None:
    """从本地 JSON 文件恢复指定会话。"""
    file_path = get_session_file(session_name)
    if not os.path.exists(file_path):
        st.error("会话不存在。")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        st.session_state.current_session = session_name
        st.session_state.messages = data.get("messages", [])
        st.session_state.workflow_step = data.get("workflow_step", "draft")
        st.session_state.user_requirement = data.get("user_requirement", "")
        st.session_state.llm_params = data.get("llm_params")
        st.session_state.llm_meta = data.get("llm_meta")
        st.session_state.reject_feedback = data.get("reject_feedback", "")
        st.session_state.feedback_result = data.get("feedback_result")
        st.session_state.final_result = data.get("final_result")
        st.session_state.outbox_events = data.get("outbox_events", [])
    except Exception as error:
        st.error(f"加载会话失败: {error}")


def delete_session(session_name: str) -> None:
    """删除会话文件；若删除的是当前会话则创建新空会话。"""
    file_path = get_session_file(session_name)
    if os.path.exists(file_path):
        os.remove(file_path)

    if session_name == st.session_state.current_session:
        st.session_state.current_session = generate_session_name()
        st.session_state.messages = []
        st.session_state.outbox_events = []
        reset_workflow()


def add_message(role: str, content: str) -> None:
    """追加一条对话日志到前端消息列表。"""
    st.session_state.messages.append({"role": role, "content": content})


def send_to_outbox(event_type: str, payload: Dict[str, Any]) -> None:
    """记录一条 outbox 事件，便于后续接入消息队列/后端服务。"""
    st.session_state.outbox_events.append(
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": event_type,
            "payload": payload,
        }
    )


def _filter_supported_kwargs(func: Callable[..., Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """根据目标函数签名筛选 kwargs，避免多传参数导致异常。"""
    signature = inspect.signature(func)
    parameters = signature.parameters

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return context

    supported: Dict[str, Any] = {}
    for name, param in parameters.items():
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            if name in context:
                supported[name] = context[name]
    return supported


def call_llm_function(function_name: str, context: Dict[str, Any]) -> Tuple[Optional[Any], Optional[str]]:
    """动态导入 llm.py 并调用指定函数，统一返回 (数据, 错误)。"""
    try:
        module = importlib.import_module(LLM_MODULE)
    except ModuleNotFoundError:
        return None, f"未找到 `{LLM_MODULE}.py`。"
    except Exception as error:
        return None, f"导入 `{LLM_MODULE}` 失败: {error}"

    target = getattr(module, function_name, None)
    if not callable(target):
        return None, f"`{LLM_MODULE}` 中未找到可调用函数 `{function_name}`。"

    try:
        kwargs = _filter_supported_kwargs(target, context)
        return target(**kwargs), None
    except TypeError as error:
        return None, f"参数不匹配 `{LLM_MODULE}.{function_name}`: {error}"
    except Exception as error:
        return None, f"调用 `{LLM_MODULE}.{function_name}` 失败: {error}"


def extract_response(raw: Any, preferred_keys: List[str]) -> Tuple[Optional[Any], Optional[str]]:
    """统一解包 llm.py 返回值，兼容 tuple/dict/raw 三种格式。"""
    value = raw

    if isinstance(value, tuple) and len(value) == 2:
        data, error = value
        if error:
            return None, str(error)
        value = data

    if isinstance(value, dict):
        if value.get("error"):
            return None, str(value.get("error"))
        for key in preferred_keys:
            if key in value:
                return value[key], None

    return value, None


def normalize_json_candidate(raw_text: str) -> List[str]:
    """从文本中提取可能的 JSON 片段（支持代码块和包裹片段）。"""
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


def parse_json(raw: Any) -> Tuple[Optional[Any], Optional[str]]:
    """把任意返回值尽量解析成 JSON(dict/list)。"""
    if isinstance(raw, (dict, list)):
        return raw, None

    if not isinstance(raw, str):
        return None, f"不支持的结果类型: {type(raw).__name__}"

    for candidate in normalize_json_candidate(raw):
        try:
            return json.loads(candidate), None
        except json.JSONDecodeError:
            pass

    return None, "返回值不是合法 JSON。"


def ensure_json_result(value: Any, source_name: str) -> Tuple[Optional[Any], Optional[str]]:
    """对 parse_json 的错误前缀化，标明来源模块。"""
    parsed, error = parse_json(value)
    if error:
        return None, f"{source_name}: {error}"
    return parsed, None


def import_llm_params() -> Tuple[Optional[Any], Optional[Dict[str, Any]], Optional[str]]:
    """调用 llm.import_param_json 并返回 (params_json, meta, error)。"""
    context = {
        "requirement": st.session_state.user_requirement,
        "session_id": st.session_state.current_session,
        "messages": st.session_state.messages,
        "workflow_step": st.session_state.workflow_step,
    }
    raw, error = call_llm_function(LLM_IMPORT_PARAM_FN, context)
    if error:
        return None, None, error

    if isinstance(raw, dict):
        meta = raw.get("meta")
    else:
        meta = None

    payload, error = extract_response(raw, ["params_json", "data", "result", "payload"])
    if error:
        return None, meta, error

    params_json, error = ensure_json_result(payload, "大模型参数")
    if error:
        return None, meta, error

    return params_json, meta, None


def submit_user_feedback(is_correct: bool, user_feedback: str = "") -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """调用 llm.handle_user_feedback 处理用户确认或驳回意见。"""
    context = {
        "requirement": st.session_state.user_requirement,
        "params_json": st.session_state.llm_params,
        "is_correct": is_correct,
        "user_feedback": user_feedback,
        "session_id": st.session_state.current_session,
        "messages": st.session_state.messages,
        "workflow_step": st.session_state.workflow_step,
    }
    raw, error = call_llm_function(LLM_FEEDBACK_FN, context)
    if error:
        return None, error

    if isinstance(raw, dict) and raw.get("error"):
        return None, str(raw.get("error"))

    if isinstance(raw, dict):
        return raw, None

    return {"raw": raw}, None


def import_final_result() -> Tuple[Optional[Any], Optional[str]]:
    """调用 llm.import_final_result 拉取最终排产结果。"""
    context = {
        "requirement": st.session_state.user_requirement,
        "params_json": st.session_state.llm_params,
        "session_id": st.session_state.current_session,
        "messages": st.session_state.messages,
        "workflow_step": st.session_state.workflow_step,
    }
    raw, error = call_llm_function(LLM_IMPORT_RESULT_FN, context)
    if error:
        return None, error

    payload, error = extract_response(raw, ["final_result", "schedule_result", "data", "result", "payload"])
    if error:
        return None, error

    result_json, error = ensure_json_result(payload, "最终结果")
    if error:
        return None, error

    return result_json, None


def import_llm_debug_state() -> Tuple[Optional[Any], Optional[str]]:
    """调用 llm.get_session_debug_state 获取当前会话调试状态。"""
    context = {"session_id": st.session_state.current_session}
    raw, error = call_llm_function(LLM_DEBUG_FN, context)
    if error:
        return None, error
    return raw, None


def render_schedule_result(result: Any) -> None:
    """渲染最终结果：优先中文可读视图，JSON 放到调试折叠区。"""
    st.subheader("最终重排产结果")

    if isinstance(result, dict):
        objective_map = {
            "minimize_makespan": "最小化总完工时间（Makespan）",
            "weighted_completion": "最小化加权完工时间",
            "minimize_weighted_completion": "最小化加权完工时间",
        }

        def _zh_job_name(name: Any) -> str:
            text = str(name)
            if text.startswith("J") and text[1:].isdigit():
                return f"J{int(text[1:]) + 1}"
            return text

        def _zh_machine_name(name: Any) -> str:
            text = str(name)
            if text.startswith("M") and text[1:].isdigit():
                return f"M{int(text[1:]) + 1}"
            return text

        instance = result.get("instance", "-")
        objective_raw = str(result.get("objective", "-"))
        objective_text = objective_map.get(objective_raw, objective_raw)
        makespan = result.get("makespan", "-")
        objective_value = result.get("objective_value", "-")
        weighted_completion = result.get("weighted_completion", "-")
        operations = result.get("operations", [])
        job_completion = result.get("job_completion_times", {})
        machine_completion = result.get("machine_completion_times", {})
        runtime_constraints = result.get("runtime_constraints", {})
        alns_stats = result.get("alns_stats", {})

        st.success("已完成排产重构并返回结果。")
        st.markdown(
            f"**实例**：`{instance}`  \n"
            f"**优化目标**：{objective_text}"
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("总完工时间", str(makespan))
        col2.metric("目标函数值", str(objective_value))
        col3.metric("加权完工值", str(round(float(weighted_completion), 3)) if isinstance(weighted_completion, (int, float)) else str(weighted_completion))
        col4.metric("工序数量", str(len(operations) if isinstance(operations, list) else 0))

        if isinstance(alns_stats, dict) and alns_stats:
            st.markdown("### 求解过程摘要")
            iter_count = alns_stats.get("iterations", "-")
            accepted_moves = alns_stats.get("accepted_moves", "-")
            improved_moves = alns_stats.get("improved_moves", "-")
            best_updates = alns_stats.get("best_updates", "-")
            st.write(
                f"本次 ALNS 共迭代 **{iter_count}** 次，"
                f"接受解 **{accepted_moves}** 次，改进解 **{improved_moves}** 次，"
                f"刷新最优解 **{best_updates}** 次。"
            )

        if isinstance(runtime_constraints, dict):
            machine_downtime = runtime_constraints.get("machine_downtime", [])
            if isinstance(machine_downtime, list) and machine_downtime:
                lines: List[str] = []
                for item in machine_downtime:
                    if isinstance(item, (list, tuple)) and len(item) == 3:
                        machine_id, start, end = item
                        try:
                            machine_label = f"M{int(machine_id) + 1}"
                        except (TypeError, ValueError):
                            machine_label = f"M{machine_id}"
                        lines.append(f"{machine_label} 停机窗口：[{start}, {end})")
                if lines:
                    st.markdown("### 生效约束（关键）")
                    st.write("；".join(lines))

        if isinstance(job_completion, dict) and job_completion:
            st.markdown("### 作业完工时间")
            jobs_table = [{"作业": _zh_job_name(job), "完工时间": value} for job, value in job_completion.items()]
            st.dataframe(jobs_table, use_container_width=True)

        if isinstance(machine_completion, dict) and machine_completion:
            st.markdown("### 机器完工时间")
            machines_table = [{"机器": _zh_machine_name(machine), "完工时间": value} for machine, value in machine_completion.items()]
            st.dataframe(machines_table, use_container_width=True)

        if isinstance(operations, list) and operations and isinstance(operations[0], dict):
            st.markdown("### 详细排程（按开始时间排序）")
            op_rows: List[Dict[str, Any]] = []
            for item in operations:
                machine_raw = item.get("machine")
                try:
                    machine_text = f"M{int(machine_raw) + 1}"
                except (TypeError, ValueError):
                    machine_text = str(machine_raw)
                op_rows.append(
                    {
                        "序号": item.get("seq_pos"),
                        "作业": _zh_job_name(item.get("job")),
                        "工序序号": item.get("operation_index"),
                        "机器": machine_text,
                        "加工时长": item.get("duration"),
                        "开始时间": item.get("start"),
                        "结束时间": item.get("end"),
                    }
                )
            op_rows.sort(key=lambda row: (row.get("开始时间") if isinstance(row.get("开始时间"), (int, float)) else 0, row.get("序号") if isinstance(row.get("序号"), (int, float)) else 0))

            # 甘特图：横轴为时间，纵轴为机器，颜色区分作业。
            machine_order = sorted(
                {str(row.get("机器", "")) for row in op_rows},
                key=lambda name: int(name[1:]) if isinstance(name, str) and name.startswith("M") and name[1:].isdigit() else 9999,
            )
            gantt_spec = {
                "mark": {"type": "bar", "cornerRadius": 2},
                "encoding": {
                    "x": {"field": "开始时间", "type": "quantitative", "title": "时间"},
                    "x2": {"field": "结束时间"},
                    "y": {"field": "机器", "type": "nominal", "title": "机器", "sort": machine_order},
                    "color": {"field": "作业", "type": "nominal", "title": "作业"},
                    "tooltip": [
                        {"field": "作业", "type": "nominal"},
                        {"field": "工序序号", "type": "quantitative"},
                        {"field": "机器", "type": "nominal"},
                        {"field": "开始时间", "type": "quantitative"},
                        {"field": "结束时间", "type": "quantitative"},
                        {"field": "加工时长", "type": "quantitative"},
                    ],
                },
                "height": 320,
            }
            st.markdown("#### 甘特图")
            st.vega_lite_chart(op_rows, gantt_spec, use_container_width=True)
            st.dataframe(op_rows, use_container_width=True)

        with st.expander("查看原始 JSON（调试）"):
            st.json(result)
        return

    if isinstance(result, list) and result and isinstance(result[0], dict):
        st.markdown("### 结果表格")
        st.dataframe(result, use_container_width=True)
        with st.expander("查看原始 JSON（调试）"):
            st.json(result)
        return

    st.warning("结果结构不标准，展示原始内容。")
    st.json(result)

init_state()

st.title("排产前端工作台")
st.caption("前端负责约束输入与确认，参数提取/校验/算法接口由 llm.py 负责。")

current_step = st.session_state.workflow_step
if current_step not in STEP_ORDER:
    current_step = "draft"
    st.session_state.workflow_step = "draft"
current_index = STEP_ORDER.index(current_step) + 1
st.progress(current_index / len(STEP_ORDER))
st.write(f"当前步骤: {STEP_TEXT[current_step]}")

with st.sidebar:
    st.subheader("会话管理")
    if st.button("新建会话", use_container_width=True):
        save_session()
        st.session_state.current_session = generate_session_name()
        st.session_state.messages = []
        st.session_state.outbox_events = []
        reset_workflow()
        st.rerun()

    if st.button("重置当前流程", use_container_width=True):
        reset_workflow()
        save_session()
        st.rerun()

    st.markdown("---")
    st.markdown("**llm接口约定**")
    st.code(
        f"{LLM_MODULE}.{LLM_IMPORT_PARAM_FN}(...)\n"
        f"{LLM_MODULE}.{LLM_FEEDBACK_FN}(...)\n"
        f"{LLM_MODULE}.{LLM_IMPORT_RESULT_FN}(...)",
        language="python",
    )

    st.markdown("---")
    st.markdown("**会话历史**")
    for session in load_sessions():
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(
                session,
                key=f"load_{session}",
                use_container_width=True,
                type="primary" if session == st.session_state.current_session else "secondary",
            ):
                load_session(session)
                st.rerun()
        with col2:
            if st.button("删", key=f"delete_{session}", use_container_width=True):
                delete_session(session)
                save_session()
                st.rerun()


tab1, tab2, tab3, tab4 = st.tabs(["1) 约束输入", "2) 参数确认", "3) 结果展示", "4) 调试日志"])

with tab1:
    st.markdown("### 输入本次调整约束")
    st.text_area(
        "约束描述",
        key="user_requirement",
        height=160,
        placeholder="例如：在当前排产任务基础上，订单A/B优先级提高，机器M2明天停机维护4小时，需要重排未来三天计划。",
    )

    if st.button("提交约束并生成参数", type="primary", use_container_width=True):
        content = st.session_state.user_requirement.strip()
        if not content:
            st.warning("请先填写约束描述。")
        else:
            add_message("user", content)
            send_to_outbox("request_param_json", {"requirement": content})
            st.session_state.workflow_step = "awaiting_llm_json"
            st.session_state.llm_params = None
            st.session_state.llm_meta = None
            st.session_state.final_result = None
            st.session_state.feedback_result = None
            # 在用户提交约束后，立即调用 llm.py 生成参数，直接进入核验环节。
            params, meta, error = import_llm_params()
            if error:
                save_session()
                st.error(f"参数自动生成失败：{error}")
                st.info("你可以在步骤2点击“重新生成参数 JSON”进行重试。")
            else:
                st.session_state.llm_params = params
                st.session_state.llm_meta = meta
                st.session_state.workflow_step = "awaiting_confirmation"
                add_message("assistant", "已自动生成参数 JSON，请核验。")
                send_to_outbox("llm_params_imported", {"params_json": params, "meta": meta, "source": "auto"})
                save_session()
                st.success("参数 JSON 已自动生成，请在步骤2核验。")

with tab2:
    st.markdown("### 参数 JSON 核验")
    st.info(f"参数由 `{LLM_MODULE}.{LLM_IMPORT_PARAM_FN}` 生成；可在此手动重试生成。")

    if st.button("重新生成参数 JSON", use_container_width=True):
        if not st.session_state.user_requirement.strip():
            st.warning("请先在步骤1输入并提交约束。")
        else:
            params, meta, error = import_llm_params()
            if error:
                st.error(error)
            else:
                st.session_state.llm_params = params
                st.session_state.llm_meta = meta
                st.session_state.workflow_step = "awaiting_confirmation"
                add_message("assistant", "已重新生成参数 JSON，请核验。")
                send_to_outbox("llm_params_imported", {"params_json": params, "meta": meta, "source": "manual_retry"})
                save_session()
                st.success("参数 JSON 重新生成成功，请确认。")

    if st.session_state.llm_params is not None:
        st.markdown("#### 参数回显")
        st.json(st.session_state.llm_params)

        if st.session_state.llm_meta:
            with st.expander("参数生成元信息(meta)"):
                st.json(st.session_state.llm_meta)

        st.markdown("#### 用户确认")
        st.text_area(
            "驳回原因（仅在“参数有误”时填写）",
            key="reject_feedback",
            height=100,
            placeholder="例如：J3工序时长错误、机器编号缺失、优先级字段不完整。",
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("参数正确，提交并自动求解", type="primary", use_container_width=True):
                result, error = submit_user_feedback(is_correct=True)
                if error:
                    st.error(error)
                else:
                    st.session_state.feedback_result = result
                    add_message("user", "参数确认通过。")
                    send_to_outbox("params_confirmed", {"params_json": st.session_state.llm_params, "llm_feedback": result})

                    dispatch_result = result.get("dispatch_result") if isinstance(result, dict) else None
                    dispatch_accepted = bool(dispatch_result.get("accepted")) if isinstance(dispatch_result, dict) else False
                    if not dispatch_accepted:
                        message = ""
                        if isinstance(dispatch_result, dict):
                            message = str(dispatch_result.get("message", "算法端未接受任务"))
                        st.session_state.workflow_step = "awaiting_confirmation"
                        save_session()
                        st.error(f"算法端提交失败：{message or '请检查 llm.py/deal.py 接口实现。'}")
                    else:
                        st.session_state.workflow_step = "awaiting_final_result"

                        # 参数确认后，自动拉取框架结果并回显到前端，避免手动导入。
                        final_result, final_error = import_final_result()
                        if final_error:
                            save_session()
                            st.warning(f"参数已提交算法端，结果暂不可用：{final_error}")
                            st.info("可在步骤3点击“刷新结果”再次拉取。")
                        else:
                            st.session_state.final_result = final_result
                            add_message("assistant", "框架已完成求解并自动回传最终结果。")
                            send_to_outbox(
                                "final_result_imported",
                                {"result": final_result, "source": "auto_after_confirm"},
                            )
                            st.session_state.workflow_step = "completed"
                            save_session()
                            st.success("参数已确认，求解完成，最终结果已自动展示。")

        with col2:
            if st.button("参数有误，要求重输", use_container_width=True):
                feedback = st.session_state.reject_feedback.strip()
                if not feedback:
                    st.warning("请先填写驳回原因。")
                else:
                    result, error = submit_user_feedback(is_correct=False, user_feedback=feedback)
                    if error:
                        st.error(error)
                    else:
                        st.session_state.feedback_result = result
                        add_message("user", f"参数驳回：{feedback}")
                        send_to_outbox(
                            "params_rejected",
                            {
                                "params_json": st.session_state.llm_params,
                                "feedback": feedback,
                                "llm_feedback": result,
                            },
                        )
                        st.session_state.workflow_step = "awaiting_llm_json"
                        st.session_state.llm_params = None
                        st.session_state.llm_meta = None
                        st.session_state.final_result = None
                        save_session()
                        st.warning("驳回意见已提交，请重新导入参数 JSON。")

    if st.session_state.feedback_result:
        with st.expander("最近一次反馈处理结果"):
            st.json(st.session_state.feedback_result)

with tab3:
    st.markdown("### 最终重排产结果（自动回显）")
    st.info(f"参数确认后会自动调用 `{LLM_MODULE}.{LLM_IMPORT_RESULT_FN}` 获取结果。")

    if st.button("刷新结果", use_container_width=True):
        if st.session_state.llm_params is None:
            st.warning("请先完成参数导入与确认。")
        else:
            result, error = import_final_result()
            if error:
                st.error(error)
            else:
                st.session_state.final_result = result
                add_message("assistant", "已刷新并拿到最终重排产结果。")
                send_to_outbox("final_result_imported", {"result": result, "source": "manual_refresh"})
                st.session_state.workflow_step = "completed"
                save_session()
                st.success("最终结果已更新。")

    if st.session_state.final_result is not None:
        render_schedule_result(st.session_state.final_result)

with tab4:
    st.markdown("### 对话日志")
    if st.session_state.messages:
        for message in st.session_state.messages:
            st.chat_message(message["role"]).write(message["content"])
    else:
        st.info("暂无日志。")

    st.markdown("### 待后端消费事件（Outbox）")
    if st.session_state.outbox_events:
        st.json(st.session_state.outbox_events)
    else:
        st.info("当前没有待处理事件。")

    st.markdown("### LLM 调试状态")
    if st.button("读取 llm 会话状态", use_container_width=True):
        debug_state, error = import_llm_debug_state()
        if error:
            st.error(error)
        else:
            st.session_state.llm_debug_state = debug_state

    if st.session_state.llm_debug_state is not None:
        st.json(st.session_state.llm_debug_state)

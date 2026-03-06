from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app_llm_config import DEAL_ALNS_CONFIG


@dataclass
class ProblemInstance:
    instance_name: str
    num_jobs: int
    num_machines: int
    jobs: List[List[Tuple[int, int]]]


@dataclass
class DecodedSolution:
    sequence: List[int]
    operations: List[Dict[str, Any]]
    job_completion_times: Dict[str, int]
    machine_completion_times: Dict[str, int]
    makespan: int
    weighted_completion: float
    objective_value: float


_INSTANCE_CACHE: Optional[ProblemInstance] = None
_RESULT_STORE: Dict[str, Dict[str, Any]] = {}


def _load_instance(instance_path: str) -> ProblemInstance:
    global _INSTANCE_CACHE
    if _INSTANCE_CACHE is not None:
        return _INSTANCE_CACHE

    path = Path(instance_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / instance_path

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    jobs: List[List[Tuple[int, int]]] = []
    for job in data["jobs"]:
        tasks = [(int(machine), int(duration)) for machine, duration in job["tasks"]]
        jobs.append(tasks)

    _INSTANCE_CACHE = ProblemInstance(
        instance_name=str(data.get("instance_name", path.stem)),
        num_jobs=int(data["num_jobs"]),
        num_machines=int(data["num_machines"]),
        jobs=jobs,
    )
    return _INSTANCE_CACHE


def _normalize_machine_id(raw_id: Any, num_machines: int) -> Optional[int]:
    if raw_id is None:
        return None

    machine_num: Optional[int] = None

    if isinstance(raw_id, str):
        text = raw_id.strip().lower()
        if not text:
            return None

        # 兼容 "machine_1" / "machine1" / "M1" / "机器1" 等常见表达。
        match = re.search(r"-?\d+", text)
        if match:
            try:
                machine_num = int(match.group())
            except (TypeError, ValueError):
                machine_num = None
        else:
            try:
                machine_num = int(text)
            except (TypeError, ValueError):
                machine_num = None
    else:
        try:
            machine_num = int(raw_id)
        except (TypeError, ValueError):
            machine_num = None

    if machine_num is None:
        return None

    # 外部输入优先按 1-based 解释（用户语义更常见）；0 保留为首台机器。
    if machine_num == 0:
        return 0
    if 1 <= machine_num <= num_machines:
        return machine_num - 1
    if 0 <= machine_num < num_machines:
        return machine_num
    return None


def _extract_machine_mentions(text: str, num_machines: int) -> List[int]:
    mentions: List[int] = []
    for match in re.findall(r"(?:机器|machine[_\s-]?)(\d+)", text, flags=re.IGNORECASE):
        machine_id = _normalize_machine_id(match, num_machines)
        if machine_id is not None:
            mentions.append(machine_id)
    return mentions


def _safe_int(value: Any, default: int) -> int:
    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        text = str(value).strip()
        if not text:
            return default
        return int(float(text))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        if isinstance(value, bool):
            return float(int(value))
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def _build_runtime_constraints(params_json: Dict[str, Any], problem: ProblemInstance) -> Dict[str, Any]:
    requirement = str(params_json.get("requirement", ""))
    objective = str(params_json.get("objective", "minimize_makespan"))
    if objective not in {"minimize_makespan", "weighted_completion", "minimize_weighted_completion"}:
        objective = "minimize_makespan"
    constraints = params_json.get("constraints", [])
    algorithm_parameters = params_json.get("algorithm_parameters", {})
    if not isinstance(algorithm_parameters, dict):
        algorithm_parameters = {}

    # 算法参数优先级：JSON 明确配置 > 全局默认配置
    iterations = _safe_int(algorithm_parameters.get("iterations", DEAL_ALNS_CONFIG.alns_iterations), DEAL_ALNS_CONFIG.alns_iterations)
    destroy_ratio = _safe_float(algorithm_parameters.get("destroy_ratio", DEAL_ALNS_CONFIG.destroy_ratio), DEAL_ALNS_CONFIG.destroy_ratio)
    initial_temperature = _safe_float(
        algorithm_parameters.get("initial_temperature", DEAL_ALNS_CONFIG.initial_temperature),
        DEAL_ALNS_CONFIG.initial_temperature,
    )
    cooling_rate = _safe_float(algorithm_parameters.get("cooling_rate", DEAL_ALNS_CONFIG.cooling_rate), DEAL_ALNS_CONFIG.cooling_rate)
    min_temperature = _safe_float(algorithm_parameters.get("min_temperature", DEAL_ALNS_CONFIG.min_temperature), DEAL_ALNS_CONFIG.min_temperature)
    rng_seed = _safe_int(algorithm_parameters.get("seed", DEAL_ALNS_CONFIG.rng_seed), DEAL_ALNS_CONFIG.rng_seed)

    if iterations < 20:
        iterations = 20
    destroy_ratio = min(max(destroy_ratio, 0.05), 0.5)
    cooling_rate = min(max(cooling_rate, 0.85), 0.9999)

    # 解析故障机器与停机窗口
    failure_keywords = ["故障", "错误", "不可用", "停机", "down", "failure", "维护"]
    text_blob = " ".join([requirement] + [str(item) for item in constraints])
    has_failure_hint = any(keyword in text_blob.lower() for keyword in failure_keywords)

    affected_machines: List[int] = []

    raw_affected_machines = algorithm_parameters.get("affected_machines", [])
    if isinstance(raw_affected_machines, (list, tuple, set)):
        for raw in raw_affected_machines:
            machine_id = _normalize_machine_id(raw, problem.num_machines)
            if machine_id is not None:
                affected_machines.append(machine_id)
    elif raw_affected_machines is not None:
        machine_id = _normalize_machine_id(raw_affected_machines, problem.num_machines)
        if machine_id is not None:
            affected_machines.append(machine_id)

    if has_failure_hint and not affected_machines:
        affected_machines.extend(_extract_machine_mentions(text_blob, problem.num_machines))

    # 明确 downtime 覆盖，优先级高于 failure_status 推导
    machine_downtime: List[Tuple[int, int, int]] = []
    explicit_downtime = algorithm_parameters.get("machine_downtime", [])
    if isinstance(explicit_downtime, list):
        for item in explicit_downtime:
            if not isinstance(item, dict):
                continue
            machine_id = _normalize_machine_id(item.get("machine"), problem.num_machines)
            if machine_id is None:
                continue
            start = _safe_int(item.get("start", 0), 0)
            end = _safe_int(item.get("end", DEAL_ALNS_CONFIG.failure_block_horizon), DEAL_ALNS_CONFIG.failure_block_horizon)
            if end > start:
                machine_downtime.append((machine_id, start, end))

    failure_status = bool(algorithm_parameters.get("failure_status", False)) or has_failure_hint
    if failure_status and not machine_downtime:
        for machine_id in sorted(set(affected_machines)):
            machine_downtime.append((machine_id, 0, DEAL_ALNS_CONFIG.failure_block_horizon))

    # 作业权重（可选）
    raw_weights = algorithm_parameters.get("job_priority", {})
    job_weights: Dict[str, float] = {}
    if isinstance(raw_weights, dict):
        for key, value in raw_weights.items():
            try:
                job_weights[str(key)] = float(value)
            except (TypeError, ValueError):
                continue

    return {
        "objective": objective,
        "iterations": iterations,
        "destroy_ratio": destroy_ratio,
        "initial_temperature": initial_temperature,
        "cooling_rate": cooling_rate,
        "min_temperature": min_temperature,
        "rng_seed": rng_seed,
        "machine_downtime": machine_downtime,
        "job_weights": job_weights,
        "algorithm_parameters": algorithm_parameters,
    }


def prepare_algorithm_payload(params_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    生成实际注入求解器的参数视图。
    该视图与 run_alns 内部生效配置保持一致，可用于前端回显与提交前核验。
    """
    if not isinstance(params_json, dict):
        raise ValueError("params_json 必须为 dict")

    problem = _load_instance(DEAL_ALNS_CONFIG.instance_path)
    runtime = _build_runtime_constraints(params_json, problem)

    machine_downtime = [
        {"machine": machine, "start": start, "end": end}
        for machine, start, end in runtime["machine_downtime"]
    ]

    prepared = {
        "task_type": str(params_json.get("task_type", "reschedule")),
        "requirement": str(params_json.get("requirement", "")),
        "objective": runtime["objective"],
        "constraints": params_json.get("constraints", []),
        "algorithm_parameters": {
            "iterations": runtime["iterations"],
            "destroy_ratio": runtime["destroy_ratio"],
            "initial_temperature": runtime["initial_temperature"],
            "cooling_rate": runtime["cooling_rate"],
            "min_temperature": runtime["min_temperature"],
            "seed": runtime["rng_seed"],
            "machine_downtime": machine_downtime,
            "job_priority": runtime["job_weights"],
        },
        "runtime_binding": {
            "instance": problem.instance_name,
            "num_jobs": problem.num_jobs,
            "num_machines": problem.num_machines,
        },
    }
    return prepared


def _create_initial_sequence(problem: ProblemInstance, rng: random.Random) -> List[int]:
    sequence: List[int] = []
    for job_index, job_ops in enumerate(problem.jobs):
        sequence.extend([job_index] * len(job_ops))
    rng.shuffle(sequence)
    return sequence


def _adjust_start_for_downtime(machine: int, start: int, duration: int, downtime_by_machine: Dict[int, List[Tuple[int, int]]]) -> int:
    current_start = start
    intervals = downtime_by_machine.get(machine, [])

    for down_start, down_end in intervals:
        # 当前工序与停机窗口重叠，则将开始时间推到窗口后。
        if current_start < down_end and (current_start + duration) > down_start:
            current_start = down_end
    return current_start


def _decode_sequence(
    problem: ProblemInstance,
    sequence: List[int],
    objective: str,
    machine_downtime: List[Tuple[int, int, int]],
    job_weights: Dict[str, float],
) -> DecodedSolution:
    job_next_op = [0] * problem.num_jobs
    job_ready_time = [0] * problem.num_jobs
    machine_ready_time = [0] * problem.num_machines

    downtime_by_machine: Dict[int, List[Tuple[int, int]]] = {m: [] for m in range(problem.num_machines)}
    for machine, start, end in machine_downtime:
        downtime_by_machine[machine].append((start, end))
    for machine in downtime_by_machine:
        downtime_by_machine[machine].sort(key=lambda item: item[0])

    operations: List[Dict[str, Any]] = []

    for position, job_index in enumerate(sequence):
        op_index = job_next_op[job_index]
        if op_index >= len(problem.jobs[job_index]):
            continue

        machine, duration = problem.jobs[job_index][op_index]
        start = max(job_ready_time[job_index], machine_ready_time[machine])
        start = _adjust_start_for_downtime(machine, start, duration, downtime_by_machine)
        end = start + duration

        operations.append(
            {
                "seq_pos": position,
                "job": f"J{job_index}",
                "operation_index": op_index,
                "machine": machine,
                "duration": duration,
                "start": start,
                "end": end,
            }
        )

        job_ready_time[job_index] = end
        machine_ready_time[machine] = end
        job_next_op[job_index] += 1

    makespan = max(job_ready_time) if job_ready_time else 0

    weighted_completion = 0.0
    for job_index, completion in enumerate(job_ready_time):
        weight = job_weights.get(f"J{job_index}", 1.0)
        weighted_completion += weight * completion

    if objective in {"minimize_weighted_completion", "weighted_completion"}:
        objective_value = weighted_completion
    else:
        objective_value = float(makespan)

    return DecodedSolution(
        sequence=list(sequence),
        operations=operations,
        job_completion_times={f"J{i}": t for i, t in enumerate(job_ready_time)},
        machine_completion_times={f"M{i}": t for i, t in enumerate(machine_ready_time)},
        makespan=makespan,
        weighted_completion=weighted_completion,
        objective_value=objective_value,
    )


def _destroy_random(sequence: List[int], remove_count: int, rng: random.Random) -> Tuple[List[int], List[int]]:
    if remove_count <= 0:
        return list(sequence), []

    indices = list(range(len(sequence)))
    rng.shuffle(indices)
    to_remove = set(indices[:remove_count])

    kept: List[int] = []
    removed: List[int] = []
    for idx, job in enumerate(sequence):
        if idx in to_remove:
            removed.append(job)
        else:
            kept.append(job)
    return kept, removed


def _destroy_segment(sequence: List[int], remove_count: int, rng: random.Random) -> Tuple[List[int], List[int]]:
    if remove_count <= 0 or remove_count >= len(sequence):
        if remove_count <= 0:
            return list(sequence), []
        return [], list(sequence)

    start = rng.randint(0, len(sequence) - remove_count)
    removed = sequence[start : start + remove_count]
    kept = sequence[:start] + sequence[start + remove_count :]
    return kept, list(removed)


def _repair_random_insert(partial: List[int], removed: List[int], rng: random.Random) -> List[int]:
    candidate = list(partial)
    for job in removed:
        insert_pos = rng.randint(0, len(candidate))
        candidate.insert(insert_pos, job)
    return candidate


def _repair_greedy_insert(
    partial: List[int],
    removed: List[int],
    problem: ProblemInstance,
    objective: str,
    machine_downtime: List[Tuple[int, int, int]],
    job_weights: Dict[str, float],
) -> List[int]:
    candidate = list(partial)
    for job in removed:
        best_seq = None
        best_obj = float("inf")
        for pos in range(len(candidate) + 1):
            trial = candidate[:pos] + [job] + candidate[pos:]
            decoded = _decode_sequence(problem, trial, objective, machine_downtime, job_weights)
            if decoded.objective_value < best_obj:
                best_obj = decoded.objective_value
                best_seq = trial
        candidate = best_seq if best_seq is not None else candidate + [job]
    return candidate


def _roulette_select(weight_map: Dict[str, float], rng: random.Random) -> str:
    items = list(weight_map.items())
    total = sum(max(value, 0.001) for _, value in items)
    pivot = rng.random() * total
    acc = 0.0
    for name, value in items:
        acc += max(value, 0.001)
        if acc >= pivot:
            return name
    return items[-1][0]


def run_alns(params_json: Dict[str, Any]) -> Dict[str, Any]:
    problem = _load_instance(DEAL_ALNS_CONFIG.instance_path)
    runtime = _build_runtime_constraints(params_json, problem)

    objective = runtime["objective"]
    iterations = runtime["iterations"]
    destroy_ratio = runtime["destroy_ratio"]
    initial_temperature = runtime["initial_temperature"]
    cooling_rate = runtime["cooling_rate"]
    min_temperature = runtime["min_temperature"]
    rng = random.Random(runtime["rng_seed"])
    machine_downtime = runtime["machine_downtime"]
    job_weights = runtime["job_weights"]

    sequence = _create_initial_sequence(problem, rng)
    current = _decode_sequence(problem, sequence, objective, machine_downtime, job_weights)
    best = current

    destroy_weights: Dict[str, float] = {
        "random": 1.0,
        "segment": 1.0,
    }
    repair_weights: Dict[str, float] = {
        "random_insert": 1.0,
        "greedy_insert": 1.0,
    }

    accepted_moves = 0
    improved_moves = 0
    best_updates = 0

    temperature = initial_temperature
    remove_count = max(1, int(len(sequence) * destroy_ratio))

    for _ in range(iterations):
        destroy_name = _roulette_select(destroy_weights, rng)
        repair_name = _roulette_select(repair_weights, rng)

        if destroy_name == "segment":
            partial, removed = _destroy_segment(current.sequence, remove_count, rng)
        else:
            partial, removed = _destroy_random(current.sequence, remove_count, rng)

        if repair_name == "greedy_insert":
            candidate_seq = _repair_greedy_insert(partial, removed, problem, objective, machine_downtime, job_weights)
        else:
            candidate_seq = _repair_random_insert(partial, removed, rng)

        candidate = _decode_sequence(problem, candidate_seq, objective, machine_downtime, job_weights)

        delta = candidate.objective_value - current.objective_value
        accepted = False

        if delta <= 0:
            accepted = True
        else:
            acceptance_prob = math.exp(-delta / max(temperature, 1e-9))
            if rng.random() < acceptance_prob:
                accepted = True

        reward = 0.0

        if accepted:
            accepted_moves += 1
            current = candidate
            reward += 0.5

            if delta < 0:
                improved_moves += 1
                reward += 1.5

            if candidate.objective_value < best.objective_value:
                best = candidate
                best_updates += 1
                reward += 3.0

        alpha = DEAL_ALNS_CONFIG.weight_alpha
        destroy_weights[destroy_name] = (1 - alpha) * destroy_weights[destroy_name] + alpha * max(reward, 0.1)
        repair_weights[repair_name] = (1 - alpha) * repair_weights[repair_name] + alpha * max(reward, 0.1)

        temperature = max(min_temperature, temperature * cooling_rate)

    return {
        "instance": problem.instance_name,
        "objective": objective,
        "objective_value": best.objective_value,
        "makespan": best.makespan,
        "weighted_completion": best.weighted_completion,
        "job_completion_times": best.job_completion_times,
        "machine_completion_times": best.machine_completion_times,
        "operations": best.operations,
        "alns_stats": {
            "iterations": iterations,
            "accepted_moves": accepted_moves,
            "improved_moves": improved_moves,
            "best_updates": best_updates,
            "final_temperature": round(temperature, 6),
            "destroy_weights": destroy_weights,
            "repair_weights": repair_weights,
        },
        "runtime_constraints": {
            "machine_downtime": machine_downtime,
            "job_weights": job_weights,
            "algorithm_parameters": runtime["algorithm_parameters"],
        },
    }


def solve_from_params(params_json: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
    result = run_alns(params_json)
    if session_id:
        _RESULT_STORE[session_id] = result
    return result


def submit_job(requirement: str, params_json: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    _ = requirement
    if not isinstance(params_json, dict):
        return {
            "accepted": False,
            "job_id": session_id,
            "message": "params_json 必须为 dict",
        }

    result = solve_from_params(params_json, session_id=session_id)
    return {
        "accepted": True,
        "job_id": session_id,
        "message": "ALNS 已执行完成",
        "objective": result.get("objective"),
        "makespan": result.get("makespan"),
        "objective_value": result.get("objective_value"),
    }


def get_job_result(session_id: str) -> Optional[Dict[str, Any]]:
    return _RESULT_STORE.get(session_id)

def get_job_result(session_id: str) -> Optional[Dict[str, Any]]:
    return _RESULT_STORE.get(session_id)



import ast
import copy
import math
from typing import Any, Dict, List, Tuple, Optional


SKIP_PARAM_NAMES = {
    "epsilon",
    "eps",
    "tiny",
    "small_constant",
    "inf",
    "nan",
}

# Very common aliases that theorist may mention loosely
PARAM_NAME_ALIASES = {
    "alpha_coef": "alpha",
    "beta_coef": "beta",
    "probability": "prob",
    "learning_rate": "lr",
    "restart_threshold": "restart_patience",
    "perturbation_steps": "perturb_steps",
}


def _literal_eval_params_dict(code: str) -> Dict[str, Any]:
    """
    Parse top-level PARAMS = {...} from Python source code using AST.
    Returns a plain dict.
    Raises ValueError if PARAMS is missing or invalid.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Code is not valid Python: {e}") from e

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "PARAMS":
                    try:
                        value = ast.literal_eval(node.value)
                    except Exception as e:
                        raise ValueError(f"PARAMS exists but is not a literal dict: {e}") from e
                    if not isinstance(value, dict):
                        raise ValueError("PARAMS must be a plain Python dict.")
                    return value

    raise ValueError("Top-level PARAMS dictionary not found.")


def _python_literal(value: Any) -> str:
    """Serialize a Python literal in source form."""
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    return repr(value)


def _replace_params_dict_in_code(code: str, new_params: Dict[str, Any]) -> str:
    """
    Replace the top-level PARAMS dict in code with a new literal dict.
    This implementation replaces the first top-level assignment to PARAMS.
    """
    lines = code.splitlines()
    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("PARAMS") and "=" in stripped:
            left = stripped.split("=", 1)[0].strip()
            if left == "PARAMS":
                start_idx = i
                break

    if start_idx is None:
        raise ValueError("Could not find top-level PARAMS assignment in code.")

    brace_balance = 0
    found_open = False

    for j in range(start_idx, len(lines)):
        line = lines[j]
        brace_balance += line.count("{")
        brace_balance -= line.count("}")
        if "{" in line:
            found_open = True
        if found_open and brace_balance == 0:
            end_idx = j
            break

    if end_idx is None:
        raise ValueError("Could not determine the end of PARAMS dict.")

    dict_lines = ["PARAMS = {"]
    for k, v in new_params.items():
        dict_lines.append(f"    {repr(k)}: {_python_literal(v)},")
    dict_lines.append("}")

    new_block = "\n".join(dict_lines)
    new_lines = lines[:start_idx] + [new_block] + lines[end_idx + 1:]
    return "\n".join(new_lines) + ("\n" if code.endswith("\n") else "")


def _is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _is_nonbool_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _normalize_name(name: str) -> str:
    return str(name).strip().lower()


def _tokenize_name(name: str) -> List[str]:
    return [tok for tok in _normalize_name(name).replace("-", "_").split("_") if tok]


def _resolve_parameter_name(requested_name: str, params: Dict[str, Any]) -> str:
    """
    Resolve a theorist-provided parameter name to a real PARAMS key.
    Strategy:
    1) exact match
    2) alias lookup
    3) normalized exact match
    4) token-overlap best match
    """
    if requested_name in params:
        return requested_name

    requested_norm = _normalize_name(requested_name)
    if requested_norm in PARAM_NAME_ALIASES:
        alias_target = PARAM_NAME_ALIASES[requested_norm]
        if alias_target in params:
            return alias_target

    norm_to_real = {_normalize_name(k): k for k in params.keys()}
    if requested_norm in norm_to_real:
        return norm_to_real[requested_norm]

    requested_tokens = set(_tokenize_name(requested_name))
    best_key = None
    best_score = 0

    for real_key in params.keys():
        real_tokens = set(_tokenize_name(real_key))
        if not real_tokens:
            continue
        overlap = len(requested_tokens & real_tokens)
        if overlap > best_score:
            best_score = overlap
            best_key = real_key

    if best_key is not None and best_score > 0:
        return best_key

    raise ValueError(f"Parameter '{requested_name}' not found in PARAMS.")


def _infer_bounds(name: str, value: Any) -> Tuple[Optional[float], Optional[float], str]:
    """
    Infer loose numeric bounds from parameter name and current value.
    Returns (lower_bound, upper_bound, bound_reason)
    """
    lowered = _normalize_name(name)

    # Probability / ratio-like parameters
    bounded_markers = {
        "ratio", "rate", "prob", "probability", "alpha", "beta",
        "momentum", "decay", "discount", "dropout", "fraction"
    }
    if any(marker in lowered for marker in bounded_markers):
        return 0.0, 1.0, "name suggests bounded float in [0, 1]"

    # Positive control parameters
    positive_markers = {
        "weight", "penalty", "coef", "coeff", "scale", "strength",
        "temperature", "threshold", "patience", "budget", "steps",
        "iters", "iterations", "count", "topk", "k", "window",
        "radius", "depth", "width", "size"
    }
    if any(marker in lowered for marker in positive_markers):
        return 0.0, None, "name suggests non-negative control parameter"

    # Fallback by value type
    if _is_nonbool_int(value):
        return 0.0, None, "integer fallback bound"
    if isinstance(value, float):
        return None, None, "unbounded float fallback"

    return None, None, "unsupported type"


def _infer_param_kind(name: str, value: Any) -> str:
    """
    Infer a richer parameter kind for logging and candidate generation.
    """
    if isinstance(value, bool):
        return "bool"

    if _is_nonbool_int(value):
        lb, ub, _ = _infer_bounds(name, value)
        if lb == 0.0 and ub is None:
            return "non_negative_int"
        return "int"

    if isinstance(value, float):
        if not math.isfinite(value):
            return "unsupported"

        lb, ub, _ = _infer_bounds(name, value)
        if lb == 0.0 and ub == 1.0:
            return "bounded_float"
        if value > 0:
            return "positive_float"
        if value == 0.0:
            return "zero_float"
        return "signed_float"

    return "unsupported"


def _clip_numeric(value: float, lower: Optional[float], upper: Optional[float]) -> float:
    if lower is not None:
        value = max(value, lower)
    if upper is not None:
        value = min(value, upper)
    return value


def _dedup_preserve_order(values: List[Any], original_value: Any) -> List[Any]:
    result = []
    seen = set()
    for v in values:
        key = repr(v)
        if key in seen:
            continue
        if v == original_value:
            continue
        seen.add(key)
        result.append(v)
    return result


def _generate_bool_candidates(value: bool) -> List[bool]:
    return [not value]


def _generate_int_candidates(value: int, lower: Optional[float], upper: Optional[float]) -> List[int]:
    """
    Richer local + mild global scan for integer parameters.
    """
    raw = [
        value - 2, value - 1, value + 1, value + 2,
        max(0, int(round(value * 0.5))),
        max(0, int(round(value * 0.75))),
        max(0, int(round(value * 1.25))),
        max(0, int(round(value * 1.5))),
    ]

    # Include a small positive default when current value is 0
    if value == 0:
        raw.extend([1, 2, 3])

    candidates = []
    for v in raw:
        if lower is not None:
            v = max(int(math.ceil(lower)), v)
        if upper is not None:
            v = min(int(math.floor(upper)), v)
        candidates.append(int(v))

    return _dedup_preserve_order(candidates, value)


def _generate_float_candidates(
    value: float,
    lower: Optional[float],
    upper: Optional[float],
    kind: str,
) -> List[float]:
    """
    Richer scan for float parameters:
    - local multiplicative perturbations
    - additive nudges
    - extra boundary-aware points for bounded floats
    """
    candidates = []

    if value > 0:
        for m in [0.5, 0.7, 0.85, 0.95, 1.05, 1.15, 1.3, 1.5]:
            candidates.append(value * m)

        step = max(abs(value) * 0.1, 1e-3)
        candidates.extend([value - step, value + step])
    else:
        # zero or signed float fallback
        candidates.extend([-1.0, -0.5, -0.1, 0.1, 0.5, 1.0])

    if kind == "bounded_float":
        # probe near edges and midpoints; useful when current value is poorly placed
        candidates.extend([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])

    clipped = []
    for v in candidates:
        if not math.isfinite(float(v)):
            continue
        v = _clip_numeric(float(v), lower, upper)
        v = round(v, 12)
        clipped.append(v)

    return _dedup_preserve_order(clipped, value)


def _generate_local_candidates(name: str, value: Any) -> Tuple[List[Any], str]:
    """
    Generate local scan candidates for a single parameter.
    Returns (candidates, rule_used)
    """
    if _normalize_name(name) in SKIP_PARAM_NAMES:
        return [], "skipped_by_name"

    kind = _infer_param_kind(name, value)
    lower, upper, bound_reason = _infer_bounds(name, value)

    if kind == "bool":
        return _generate_bool_candidates(value), "bool_flip"

    if kind in {"int", "non_negative_int"}:
        return _generate_int_candidates(int(value), lower, upper), f"int_scan[{bound_reason}]"

    if kind in {"bounded_float", "positive_float", "zero_float", "signed_float"}:
        return _generate_float_candidates(float(value), lower, upper, kind), f"float_scan[{bound_reason}]"

    return [], "unsupported_type"


def _build_scan_log(
    requested_parameter_name: str,
    resolved_parameter_name: str,
    old_value: Any,
    param_kind: str,
    candidates: List[Any],
    rule_used: str,
    skip_reason: str = "",
) -> str:
    """
    Build a more informative human-readable scan log.
    """
    lines = []
    lines.append("=== Local Parameter Scan Log ===")
    lines.append(f"requested_parameter_name: {requested_parameter_name}")
    lines.append(f"resolved_parameter_name: {resolved_parameter_name}")
    lines.append(f"original_value: {old_value!r}")
    lines.append(f"parameter_kind: {param_kind}")
    lines.append(f"rule_used: {rule_used}")

    if skip_reason:
        lines.append("status: skipped")
        lines.append(f"reason: {skip_reason}")
    else:
        lines.append("status: generated_variants")
        lines.append(f"num_candidates: {len(candidates)}")
        lines.append("candidate_values:")
        for i, v in enumerate(candidates, 1):
            lines.append(f"  {i}. {v!r}")

    return "\n".join(lines) + "\n"


def generate_param_scan_variants(
    base_code: str,
    parameter_name: str,
    *,
    max_variants: int = 8,
) -> Dict[str, Any]:
    """
    Generate code variants by locally scanning one entry in top-level PARAMS.

    Returns a dict:
    {
        "requested_parameter_name": ...,
        "parameter_name": ...,              # resolved real PARAMS key
        "old_value": ...,
        "param_kind": ...,
        "candidates": [...],
        "rule_used": ...,
        "scan_log": "...",
        "variants": [
            {
                "code": ...,
                "parameter_name": ...,
                "old_value": ...,
                "new_value": ...,
                "mutation_note": ...
            },
            ...
        ]
    }
    """
    params = _literal_eval_params_dict(base_code)
    resolved_name = _resolve_parameter_name(parameter_name, params)

    old_value = params[resolved_name]
    param_kind = _infer_param_kind(resolved_name, old_value)

    candidates, rule_used = _generate_local_candidates(resolved_name, old_value)

    if not candidates:
        scan_log = _build_scan_log(
            requested_parameter_name=parameter_name,
            resolved_parameter_name=resolved_name,
            old_value=old_value,
            param_kind=param_kind,
            candidates=[],
            rule_used=rule_used,
            skip_reason="no valid local candidates could be generated for this parameter type/value",
        )
        return {
            "requested_parameter_name": parameter_name,
            "parameter_name": resolved_name,
            "old_value": old_value,
            "param_kind": param_kind,
            "candidates": [],
            "rule_used": rule_used,
            "scan_log": scan_log,
            "variants": [],
        }

    candidates = candidates[:max_variants]
    scan_log = _build_scan_log(
        requested_parameter_name=parameter_name,
        resolved_parameter_name=resolved_name,
        old_value=old_value,
        param_kind=param_kind,
        candidates=candidates,
        rule_used=rule_used,
        skip_reason="",
    )

    variants = []
    for new_value in candidates:
        new_params = copy.deepcopy(params)
        new_params[resolved_name] = new_value
        new_code = _replace_params_dict_in_code(base_code, new_params)

        variants.append({
            "code": new_code,
            "parameter_name": resolved_name,
            "old_value": old_value,
            "new_value": new_value,
            "mutation_note": (
                f"Local scan on PARAMS['{resolved_name}']: {old_value!r} -> {new_value!r}"
            ),
        })

    return {
        "requested_parameter_name": parameter_name,
        "parameter_name": resolved_name,
        "old_value": old_value,
        "param_kind": param_kind,
        "candidates": candidates,
        "rule_used": rule_used,
        "scan_log": scan_log,
        "variants": variants,
    }
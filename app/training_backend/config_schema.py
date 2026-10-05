from __future__ import annotations

from typing import Any


def _coerce_scalar(value: Any, value_type: str, name: str) -> Any:
    if value is None:
        return None
    try:
        if value_type == "int":
            if isinstance(value, bool):
                raise ValueError
            return int(value)
        if value_type == "float":
            if isinstance(value, bool):
                raise ValueError
            return float(value)
        if value_type == "str":
            return str(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be {value_type}.") from exc
    return value


def compile_config_values(
    values: dict[str, Any],
    schema: list[dict[str, Any]],
    *,
    require_editable: bool,
) -> list[str]:
    fields = {field["name"]: field for field in schema}
    unknown = sorted(set(values) - set(fields))
    if unknown:
        raise ValueError(f"Unknown training configuration fields: {unknown}")

    argv: list[str] = []
    for name in sorted(values):
        field = fields[name]
        if require_editable and not field.get("editable", False):
            raise ValueError(f"{name} is read-only in the training application.")
        value = values[name]
        option_strings = list(field.get("option_strings") or [])
        if not option_strings:
            raise ValueError(f"{name} has no command-line option.")

        if field["type"] == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be bool.")
            mode = field.get("boolean_mode")
            if mode == "boolean_optional":
                preferred = (
                    next(
                        (
                            option
                            for option in option_strings
                            if option.startswith("--no-")
                        ),
                        None,
                    )
                    if not value
                    else next(
                        (
                            option
                            for option in option_strings
                            if not option.startswith("--no-")
                        ),
                        None,
                    )
                )
                if preferred:
                    argv.append(preferred)
            elif (mode == "store_true" and value) or (
                mode == "store_false" and not value
            ):
                argv.append(option_strings[0])
            elif mode == "value":
                argv.extend([option_strings[0], str(value).lower()])
            continue

        option = next(
            (item for item in option_strings if item.startswith("--")),
            option_strings[0],
        )
        if value is None:
            continue
        if field["type"] == "list":
            if not isinstance(value, list):
                raise ValueError(f"{name} must be a list.")
            list_values = [
                _coerce_scalar(item, field.get("item_type") or "str", name)
                for item in value
            ]
            argv.extend([option, *(str(item) for item in list_values)])
            continue

        coerced = _coerce_scalar(value, field["type"], name)
        choices = field.get("choices")
        if choices is not None and coerced not in choices:
            raise ValueError(f"{name} must be one of {choices}.")
        argv.extend([option, str(coerced)])
    return argv


def compile_overrides(
    overrides: dict[str, Any], schema: list[dict[str, Any]]
) -> list[str]:
    return compile_config_values(overrides, schema, require_editable=True)

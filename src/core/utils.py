import json
import re
from typing import Any, Dict, Iterable, List, Sequence, TypeVar


T = TypeVar("T")


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def dedupe_keep_order(items: Sequence[T]) -> List[T]:
    seen = set()
    output: List[T] = []
    for item in items:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def take_n(items: Iterable[T], n: int) -> List[T]:
    out: List[T] = []
    for item in items:
        if len(out) >= n:
            break
        out.append(item)
    return out


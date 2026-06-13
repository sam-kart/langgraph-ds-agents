import json
import re
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def extract_json_object(text: str) -> str:
    """Extract the first complete-looking JSON object from model output."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model response")
    candidate = cleaned[start:end + 1]
    json.loads(candidate)
    return candidate


def parse_model(text: str, model_type: type[T]) -> T:
    return model_type.model_validate_json(extract_json_object(text))

import json
import math
import re
from dataclasses import dataclass

# A whole text that is one fenced block: ```json ... ``` or ``` ... ```.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


# Frozen: a parse result can't be changed after it is created.
@dataclass(frozen=True)
class ParsedAnswer:
    ok: bool  # a usable answer was found and its types are valid
    answer: int | float | str | None
    answerable: bool | None
    clean_format: bool  # the whole reply was exactly one JSON object, with no fence or extra text
    problem: str | None  # why parsing failed; None when ok


def _fail(problem: str) -> ParsedAnswer:
    return ParsedAnswer(ok=False, answer=None, answerable=None, clean_format=False, problem=problem)


def _missing_keys(obj: dict) -> str:
    missing = [k for k in ("answer", "answerable") if k not in obj]
    return f"missing key(s): {', '.join(missing)}"


# Check the types of an object that has both keys. Values are returned exactly as sent, never converted.
def _validate(obj: dict, clean_format: bool) -> ParsedAnswer:
    answerable, answer = obj["answerable"], obj["answer"]
    # type() is not isinstance(): bool is a subclass of int, and a bool must not pass as a number.
    if type(answerable) is not bool:
        return _fail(f"answerable must be true or false, got {type(answerable).__name__}")
    if type(answer) is float:
        if not math.isfinite(answer):  # json.loads accepts NaN and Infinity, so check explicitly
            return _fail("answer must be a finite number, got NaN or infinity")
    elif answer is not None and type(answer) not in (int, str):
        return _fail(f"answer must be a number, string or null, got {type(answer).__name__}")
    return ParsedAnswer(ok=True, answer=answer, answerable=answerable, clean_format=clean_format, problem=None)


def _parse(text) -> ParsedAnswer:
    if not isinstance(text, str):
        return _fail(f"input is not a string (got {type(text).__name__})")
    stripped = text.strip()
    if not stripped:
        return _fail("empty text")

    problem = "no JSON object with 'answer' and 'answerable' found"

    # Step 1: the whole text is one JSON object. Only this path can be clean_format.
    try:
        obj = json.loads(stripped)
    except ValueError:
        pass
    else:
        if isinstance(obj, dict) and "answer" in obj and "answerable" in obj:
            return _validate(obj, clean_format=True)
        problem = _missing_keys(obj) if isinstance(obj, dict) else "JSON is not an object"

    # Step 2: the whole text is one markdown code fence around a JSON object.
    fence = _FENCE_RE.fullmatch(stripped)
    if fence:
        try:
            obj = json.loads(fence.group(1))
        except ValueError:
            pass
        else:
            if isinstance(obj, dict) and "answer" in obj and "answerable" in obj:
                return _validate(obj, clean_format=False)

    # Step 3: look for a JSON object anywhere in the text. The first one with both keys is the one judged;
    # if its values are invalid that is a failure, we do not move on to a later object.
    decoder = json.JSONDecoder()
    for i, ch in enumerate(stripped):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(stripped, i)
        except ValueError:
            continue
        if "answer" in obj and "answerable" in obj:
            return _validate(obj, clean_format=False)
    return _fail(problem)


# Turn a model's final message into a ParsedAnswer. Never raises, whatever the input is.
def parse_final_answer(text) -> ParsedAnswer:
    try:
        return _parse(text)
    except Exception as e:  # e.g. RecursionError from absurdly nested JSON
        return _fail(f"could not parse: {type(e).__name__}")

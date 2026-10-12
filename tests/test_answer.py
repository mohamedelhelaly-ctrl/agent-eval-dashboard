import dataclasses

import pytest

from agent_eval.answer import ParsedAnswer, parse_final_answer


def good(text, answer, answerable, clean):
    r = parse_final_answer(text)
    assert r == ParsedAnswer(ok=True, answer=answer, answerable=answerable, clean_format=clean, problem=None)
    assert type(r.answer) is type(answer)  # 1.0 must not turn into 1, "42" must not turn into 42


def bad(text, *words):
    r = parse_final_answer(text)
    assert (r.ok, r.answer, r.answerable, r.clean_format) == (False, None, None, False)
    assert r.problem
    for w in words:
        assert w in r.problem
    return r


# ---------- clean JSON ----------

@pytest.mark.parametrize("text,answer,answerable", [
    ('{"answer": 500, "answerable": true}', 500, True),
    ('{"answer": 7.14, "answerable": true}', 7.14, True),
    ('{"answer": "Pro", "answerable": true}', "Pro", True),
    ('{"answer": 0, "answerable": true}', 0, True),
    ('{"answer": null, "answerable": false}', None, False),
    ('  \n{"answerable": true, "answer": 3}\n\n', 3, True),  # key order and surrounding whitespace don't matter
    ('{"answer": 3, "answerable": true, "reason": "extra keys are ignored"}', 3, True),
])
def test_clean_json(text, answer, answerable):
    good(text, answer, answerable, clean=True)


# ---------- fenced JSON ----------

@pytest.mark.parametrize("text", [
    '```json\n{"answer": 5, "answerable": true}\n```',
    '```\n{"answer": 5, "answerable": true}\n```',
    '```JSON\n{"answer": 5, "answerable": true}\n```',
    '  ```json {"answer": 5, "answerable": true} ```  ',
])
def test_fenced_json_is_ok_but_not_clean(text):
    good(text, 5, True, clean=False)


# ---------- prose around the object ----------

@pytest.mark.parametrize("text", [
    'Here is the result: {"answer": 5, "answerable": true}',
    '{"answer": 5, "answerable": true} Hope that helps!',
    'Sure.\n{"answer": 5, "answerable": true}\nLet me know if you need more.',
    'Thinking {not json} first... {"answer": 5, "answerable": true}',  # an earlier brace pair is skipped
    '{"answer": 5, "answerable": true} }',  # a stray closing brace after a valid object
    '```json\n{"answer": 5, "answerable": true}\n```\nThat is the answer.',  # fence plus text after it
])
def test_prose_wrapped_json_is_ok_but_not_clean(text):
    good(text, 5, True, clean=False)


def test_nested_object_inside_another_is_found():
    good('{"result": {"answer": 5, "answerable": true}}', 5, True, clean=False)


# ---------- two objects ----------

def test_two_objects_first_valid_wins():
    good('{"answer": 1, "answerable": true} then {"answer": 2, "answerable": true}', 1, True, clean=False)


def test_first_object_with_both_keys_is_judged_even_if_invalid():
    # We do not skip an invalid object to find a later valid one; that would hide the model's mistake.
    bad('{"answer": 1, "answerable": "yes"} {"answer": 2, "answerable": true}', "answerable")


def test_objects_without_both_keys_are_skipped():
    good('{"foo": 1} {"answer": 2} {"answer": 3, "answerable": false}', 3, False, clean=False)


# ---------- answerable ----------

def test_null_answer_with_answerable_false():
    good('{"answer": null, "answerable": false}', None, False, clean=True)


@pytest.mark.parametrize("value", ['"true"', '"false"', "1", "0", "null", "[]"])
def test_answerable_must_be_a_real_bool(value):
    bad('{"answer": 5, "answerable": %s}' % value, "answerable")


# ---------- answer types ----------

@pytest.mark.parametrize("value", ["true", "false"])
def test_bool_answer_is_rejected(value):
    bad('{"answer": %s, "answerable": true}' % value, "answer", "bool")


@pytest.mark.parametrize("value", ["[1, 2]", '{"n": 1}'])
def test_list_or_object_answer_is_rejected(value):
    bad('{"answer": %s, "answerable": true}' % value, "answer")


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_nan_and_infinity_are_rejected(value):
    bad('{"answer": %s, "answerable": true}' % value, "finite")


def test_values_are_not_normalized():
    good('{"answer": "42", "answerable": true}', "42", True, clean=True)  # string stays a string
    good('{"answer": 1.0, "answerable": true}', 1.0, True, clean=True)  # float stays a float
    good('{"answer": 12345678901234567890, "answerable": true}', 12345678901234567890, True, clean=True)
    good('{"answer": " 7% ", "answerable": true}', " 7% ", True, clean=True)  # no cleaning of text either
    good('{"answer": "", "answerable": true}', "", True, clean=True)


# ---------- missing keys and bad shapes ----------

@pytest.mark.parametrize("text,missing", [
    ('{"answer": 5}', "answerable"),
    ('{"answerable": true}', "answer"),
    ('{}', "answer"),
    ('Result: {"answer": 5}', "answerable"),
])
def test_missing_key_is_named(text, missing):
    r = bad(text)
    assert r.problem and (missing in r.problem or "found" in r.problem)


def test_missing_key_message_is_specific_when_the_whole_text_is_json():
    bad('{"answer": 5}', "missing", "answerable")


@pytest.mark.parametrize("text", [
    "", "   ", "\n\t", "no json here at all", "42", "true", "null", '"just a string"',
    '{"answer": 5, "answerable": true',  # unterminated
    "{'answer': 5, 'answerable': true}",  # single quotes are not JSON
    '[1, 2, 3]', '"```json"',
])
def test_unparseable_or_wrong_shape_fails_cleanly(text):
    bad(text)


def test_json_list_of_objects_is_not_an_answer_object_but_inner_object_is_found():
    # The whole text is a list, so step 1 fails; step 3 then finds the object inside it.
    good('[{"answer": 5, "answerable": true}]', 5, True, clean=False)


# ---------- never raises ----------

@pytest.mark.parametrize("value", [None, 5, 1.5, True, [], {}, b'{"answer": 1, "answerable": true}', object()])
def test_non_string_input_fails_without_raising(value):
    r = bad(value, "not a string")
    assert type(value).__name__ in r.problem


def test_absurdly_nested_input_does_not_raise():
    bad("[" * 100_000 + "]" * 100_000)
    bad("{" * 50_000)
    bad('{"answer": ' + "[" * 100_000)


# ---------- clean_format ----------

def test_clean_format_only_for_exactly_one_json_object():
    assert parse_final_answer('{"answer": 1, "answerable": true}').clean_format is True
    assert parse_final_answer('\n  {"answer": 1, "answerable": true}  \n').clean_format is True
    assert parse_final_answer('```json\n{"answer": 1, "answerable": true}\n```').clean_format is False
    assert parse_final_answer('Answer: {"answer": 1, "answerable": true}').clean_format is False
    assert parse_final_answer('{"answer": 1, "answerable": true} thanks').clean_format is False
    assert parse_final_answer('{"answer": 1, "answerable": true}{"answer": 2, "answerable": true}').clean_format is False


def test_invalid_results_are_never_clean():
    assert parse_final_answer('{"answer": true, "answerable": true}').clean_format is False


# ---------- the dataclass ----------

def test_parsed_answer_is_frozen():
    r = parse_final_answer('{"answer": 1, "answerable": true}')
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.answer = 2

import inspect
import os
from softassertion.analysis.code_extractor import extract_code_block
from softassertion.analysis.function_finder import find_unstable_calls
from softassertion.engine.fuzz_runner import run_fuzzing_on_calls

# Internal state
_fuzz_context = {
    "filename": None,
    "start_line": None,
    "end_line": None,
}

def start_fuzz():
    frame = inspect.stack()[1]
    _fuzz_context["filename"] = frame.filename
    _fuzz_context["start_line"] = frame.lineno

def end_fuzz():
    frame = inspect.stack()[1]
    _fuzz_context["end_line"] = frame.lineno

    path = _fuzz_context["filename"]
    start = _fuzz_context["start_line"]
    end = _fuzz_context["end_line"]

    if path is None or start is None:
        raise RuntimeError("start_fuzz() must be called before end_fuzz().")

    code_snippet = extract_code_block(path, start, end)
    print("[SoftAssertionFuzzer] Extracted fuzzing region:")
    print(code_snippet)

    calls = find_unstable_calls(code_snippet)
    print("[SoftAssertionFuzzer] Found Unstable Function Calls:")
    for c in calls:
        print(" -", c)

    run_fuzzing_on_calls(calls)
    return code_snippet

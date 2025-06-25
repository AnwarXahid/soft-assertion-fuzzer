import torch
import time
import sys
import itertools
import os
import json
from datetime import datetime
from colorama import Fore, Style, init as colorama_init

from softassertion.fuzzing.history import OscillationTracker
from softassertion.utils.config import load_default_config
from softassertion.engine.input_selector import get_input_generator
from softassertion.engine.oracle_selector import get_oracle_for_function
from change_direction import (
    get_change_direction,
    get_change_direction_for_binary_operand,
)

spinner = itertools.cycle(["|", "/", "-", "\\"])
colorama_init(autoreset=True)


def run_fuzzing_on_calls(call_names):
    config = load_default_config()
    trace_entries = []
    summary = []

    # Step 2: ScanForUnstableFunctions(p, D)
    if config.get("enable_parallel", False):
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for call in call_names:
                futures.append(executor.submit(run_fuzz_for_one_call, call, config, trace_entries, summary))
            for future in concurrent.futures.as_completed(futures):
                future.result()
    else:
        for call in call_names:
            run_fuzz_for_one_call(call, config, trace_entries, summary)

    save_summary_and_trace(summary, trace_entries)


def run_fuzz_for_one_call(call, config, trace_entries, summary):
    print(f"{Fore.YELLOW}[FuzzRunner] Running soft assertion fuzzing for: {call}{Style.RESET_ALL}")
    generator = get_input_generator(call)
    oracle = get_oracle_for_function(call)

    if generator is None or oracle is None:
        print(f"[FuzzRunner] Skipped {call} (unsupported or missing components)")
        return

    time_budget = config.get("timeout", 10)
    stop_on_failure = config.get("stop_on_failure", True)
    model_type = config.get("model_type", "14_by_14")

    try:
        # Step 4: GenerateInitialInput()
        input_config = {k: v for k, v in config.items() if k in generator.__code__.co_varnames}
        inputs = generator(**input_config)
        if not isinstance(inputs, tuple):
            inputs = (inputs,)

        for tensor in inputs:
            tensor.requires_grad_(True)

        # Step 5: H ← ∅
        osc_tracker = OscillationTracker(threshold=config.get("oscillation_threshold", 3))
        start_time = time.time()

        # Step 6: while t < Timeout
        while True:
            elapsed = time.time() - start_time
            sys.stdout.write(f"\r[FuzzRunner] Fuzzing {call} {next(spinner)}")
            sys.stdout.flush()

            if elapsed > time_budget:
                print(f"{Fore.CYAN}\n[FuzzRunner] ⏱️ Timeout: {call} exceeded {time_budget} seconds{Style.RESET_ALL}")
                break

            # Step 7: v ← Execute(x, p, f, entry)
            if call in torch.nn.functional.__dict__:
                fn = getattr(torch.nn.functional, call)
            elif hasattr(torch, call):
                fn = getattr(torch, call)
            else:
                print(f"[FuzzRunner] Unsupported function: {call}")
                break

            try:
                output = fn(*inputs)
            except Exception as e:
                print(f"\n[FuzzRunner] Error in function execution: {e}")
                break

            # Step 8: signal ← SoftAssertion(A, v, f)
            if not oracle(output):
                print(f"{Fore.RED}\n[FuzzRunner] ❌ Bug triggered by input:\n{inputs}{Style.RESET_ALL}")
                os.makedirs("experiments/logs", exist_ok=True)
                log_path = f"experiments/logs/log_{call}.txt"
                with open(log_path, "w") as log:
                    log.write(f"[BUG FOUND] Function: {call}\n")
                    log.write(f"Timestamp: {datetime.now()}\n")
                    log.write(f"Input:\n{inputs}\n")
                    log.write(f"Elapsed Time: {elapsed:.3f} seconds\n")

                summary.append({"function": call, "status": "FAIL", "time": elapsed})
                trace_entries.append({
                    "call": call,
                    "input": str(inputs),
                    "status": "FAIL",
                    "timestamp": str(datetime.now()),
                    "elapsed": elapsed
                })

                if stop_on_failure:
                    return
                else:
                    break

            # Step 15: Δx ← AutoDiff(x, f, signal)
            # === Direction-based mutation with oscillation handling ===
            if len(inputs) == 1:
                direction = get_change_direction(inputs[0], call, model_type=model_type)
                delta = direction * 50 / (torch.abs(inputs[0]) + 1e-6)
                osc_tracker.update(delta)

                # Step 16–17: x′ ← ConstraintSolving(x, Δx, H)
                if osc_tracker.should_solve():
                    midpoint = osc_tracker.midpoint(inputs[0])
                    inputs = (midpoint.detach(),)
                    osc_tracker.reset()
                else:
                    inputs = (inputs[0] + delta,)

            elif len(inputs) == 2:
                direction = get_change_direction_for_binary_operand(inputs[0], inputs[1], call, model_type=model_type)
                delta1 = direction * 50 / (torch.abs(inputs[0]) + 1e-6)
                delta2 = direction * 50 / (torch.abs(inputs[1]) + 1e-6)
                osc_tracker.update(delta1 + delta2)

                if osc_tracker.should_solve():
                    midpoint = osc_tracker.midpoint(inputs[0] + inputs[1])
                    mid = midpoint / 2
                    inputs = (mid.detach(), mid.detach())
                    osc_tracker.reset()
                else:
                    inputs = (
                        inputs[0] + delta1,
                        inputs[1] + delta2,
                    )

            for tensor in inputs:
                tensor.requires_grad_(True)

        print(f"{Fore.GREEN}\n[FuzzRunner] ✅ Done: No failure found for: {call}{Style.RESET_ALL}")
        os.makedirs("experiments/logs", exist_ok=True)
        with open(f"experiments/logs/log_{call}.txt", "w") as log:
            log.write(f"[NO BUG] Function: {call}\n")
            log.write(f"Timestamp: {datetime.now()}\n")
            log.write(f"Elapsed Time: {elapsed:.3f} seconds\n")

        summary.append({"function": call, "status": "PASS", "time": elapsed})
        trace_entries.append({
            "call": call,
            "input": str(inputs),
            "status": "PASS",
            "timestamp": str(datetime.now()),
            "elapsed": elapsed
        })

    except Exception as e:
        print(f"{Fore.MAGENTA}\n[FuzzRunner] Error during fuzzing {call}: {e}{Style.RESET_ALL}")


def save_summary_and_trace(summary, trace_entries):
    os.makedirs("experiments/logs", exist_ok=True)
    os.makedirs("experiments/reports", exist_ok=True)

    with open("experiments/logs/summary_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open("experiments/logs/trace.log", "w") as f:
        for entry in trace_entries:
            f.write(json.dumps(entry) + "\n")

    # Case study summary stats
    total = len(summary)
    failed = sum(1 for x in summary if x["status"] == "FAIL")
    passed = total - failed
    avg_time = sum(x["time"] for x in summary) / total if total > 0 else 0
    fail_rate = failed / total if total > 0 else 0.0

    summary_stats = {
        "Total Functions": total,
        "Passed": passed,
        "Failed": failed,
        "Average Time (sec)": round(avg_time, 3),
        "Failure Rate": round(fail_rate * 100, 2)
    }

    with open("experiments/reports/case_study_summary.json", "w") as f:
        json.dump(summary_stats, f, indent=2)

    print("\n📊 Case Study Summary:")
    for k, v in summary_stats.items():
        print(f"  {k}: {v}")

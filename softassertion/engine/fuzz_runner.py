import yaml
import torch
import time
import sys
import itertools
import os
from datetime import datetime

from softassertion.fuzzing.history import OscillationTracker
from softassertion.utils.config import load_default_config
from softassertion.engine.input_selector import get_input_generator
from softassertion.engine.oracle_selector import get_oracle_for_function
from softassertion.utils.unstable_func_enum import UnstableFuncEnum
from change_direction import (
    get_change_direction,
    get_change_direction_for_binary_operand,
)

spinner = itertools.cycle(["|", "/", "-", "\\"])


def run_fuzzing_on_calls(call_names):
    for call in call_names:
        print(f"[FuzzRunner] Running soft assertion fuzzing for: {call}")
        generator = get_input_generator(call)
        oracle = get_oracle_for_function(call)

        if generator is None or oracle is None:
            print(f"[FuzzRunner] Skipped {call} (unsupported or missing components)")
            continue

        config = load_default_config()
        time_budget = config.get("max_fuzz_time_seconds", 10)
        stop_on_failure = config.get("stop_on_failure", True)
        model_type = config.get("model_type", "14_by_14")

        try:
            input_config = {k: v for k, v in config.items() if k in generator.__code__.co_varnames}
            inputs = generator(**input_config)
            if not isinstance(inputs, tuple):
                inputs = (inputs,)

            for tensor in inputs:
                tensor.requires_grad_(True)

            osc_tracker = OscillationTracker(threshold=config.get("oscillation_threshold", 3))

            start_time = time.time()

            while True:
                elapsed = time.time() - start_time
                sys.stdout.write(f"\r[FuzzRunner] Fuzzing {call} {next(spinner)}")
                sys.stdout.flush()

                if elapsed > time_budget:
                    print(f"\n[FuzzRunner] ⏱️ Timeout: {call} exceeded {time_budget} seconds")
                    break

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

                if not oracle(output):
                    print(f"\n[FuzzRunner] ❌ Bug triggered by input:\n{inputs}")
                    os.makedirs("experiments/logs", exist_ok=True)
                    with open(f"experiments/logs/log_{call}.txt", "w") as log:
                        log.write(f"[BUG FOUND] Function: {call}\n")
                        log.write(f"Timestamp: {datetime.now()}\n")
                        log.write(f"Input:\n{inputs}\n")
                        log.write(f"Elapsed Time: {elapsed:.3f} seconds\n")
                    if stop_on_failure:
                        return
                    else:
                        break


                    # === Direction-based mutation with oscillation handling ===
                    if len(inputs) == 1:
                        direction = get_change_direction(inputs[0], call, model_type=model_type)
                        delta = direction * 50 / (torch.abs(inputs[0]) + 1e-6)
                        osc_tracker.update(delta)

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
                            inputs = (mid.detach(), mid.detach())  # simple heuristic
                            osc_tracker.reset()
                        else:
                            inputs = (
                                inputs[0] + delta1,
                                inputs[1] + delta2,
                            )

                for tensor in inputs:
                    tensor.requires_grad_(True)

            print(f"\n[FuzzRunner] ✅ Done: No failure found for: {call}")
            os.makedirs("experiments/logs", exist_ok=True)
            with open(f"experiments/logs/log_{call}.txt", "w") as log:
                log.write(f"[NO BUG] Function: {call}\n")
                log.write(f"Timestamp: {datetime.now()}\n")
                log.write(f"Elapsed Time: {elapsed:.3f} seconds\n")

        except Exception as e:
            print(f"\n[FuzzRunner] Error during fuzzing {call}: {e}")


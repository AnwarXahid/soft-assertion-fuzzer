import argparse
import runpy
import sys
from softassertion.utils.config import set_override_config_path


def main():
    parser = argparse.ArgumentParser(description="Soft Assertion Fuzzer CLI")

    parser.add_argument("script", type=str, help="Path to the script to run.")
    parser.add_argument("--config", type=str, help="Optional path to custom YAML config.")
    parser.add_argument("--timeout", type=int, help="Override timeout in seconds.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")

    args = parser.parse_args()

    if args.config:
        set_override_config_path(args.config)

    if args.timeout:
        from softassertion.utils.config import set_runtime_override
        set_runtime_override("timeout", args.timeout)

    if args.verbose:
        print(f"[SoftAssertionCLI] Running script: {args.script}")
        if args.config:
            print(f"[SoftAssertionCLI] Using config: {args.config}")
        if args.timeout:
            print(f"[SoftAssertionCLI] Overriding timeout: {args.timeout}s")

    runpy.run_path(args.script, run_name="__main__")


if __name__ == "__main__":
    main()

import runpy
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m softassertion.cli path/to/script.py")
        sys.exit(1)

    script_path = sys.argv[1]
    print(f"[SoftAssertionCLI] Running script: {script_path}")
    runpy.run_path(script_path, run_name="__main__")

if __name__ == "__main__":
    main()

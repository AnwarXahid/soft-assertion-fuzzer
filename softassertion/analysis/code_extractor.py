def extract_code_block(filepath: str, start: int, end: int) -> str:
    with open(filepath, "r") as f:
        lines = f.readlines()[start:end - 1]
    dedented = [line.lstrip() for line in lines]
    return "".join(dedented)

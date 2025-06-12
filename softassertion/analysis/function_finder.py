import ast
from typing import List

UNSTABLE_FUNCTIONS = {
    "exp", "log", "sqrt", "relu", "softmax", "cosine_similarity", "matmul",
    "sigmoid", "tanh", "sinh", "acos", "conv2d", "mean", "sum", "square"
}

class UnstableFunctionVisitor(ast.NodeVisitor):
    def __init__(self):
        self.calls = []

    def visit_Call(self, node):
        func_name = self._get_func_name_recursive(node.func)
        if func_name:
            short_name = func_name.split(".")[-1]
            if short_name in UNSTABLE_FUNCTIONS:
                self.calls.append(short_name)
        self.generic_visit(node)

    def _get_func_name_recursive(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            base = self._get_func_name_recursive(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return None

def find_unstable_calls(code: str) -> List[str]:
    try:
        tree = ast.parse(code)
        visitor = UnstableFunctionVisitor()
        visitor.visit(tree)
        return visitor.calls
    except SyntaxError:
        return []

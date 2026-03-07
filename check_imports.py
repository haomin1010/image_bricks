
import os
import ast

def check_file(filepath):
    with open(filepath, 'r') as f:
        try:
            tree = ast.parse(f.read())
        except Exception as e:
            # print(f"Error parsing {filepath}: {e}")
            return

    has_math_import = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'math':
                    has_math_import = True
        elif isinstance(node, ast.ImportFrom):
            if node.module == 'math':
                # from math import ...
                pass

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == 'math':
            # Check if it's a usage, not an import
            if not has_math_import:
                # Check if it is inside a function where math is imported locally
                # This is hard to check with simple AST walk.
                # But let's just print it.
                print(f"Possible missing import in {filepath} at line {node.lineno}")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'math':
             if not has_math_import:
                print(f"Possible missing import in {filepath} at line {node.lineno}: math.{node.attr}")

for root, dirs, files in os.walk("VAGEN"):
    for file in files:
        if file.endswith(".py"):
            check_file(os.path.join(root, file))

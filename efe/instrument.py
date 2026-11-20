"""Apply the two collection hooks to the client or simulator module during an EFE image build."""
import argparse
import ast
from pathlib import Path


def instrument(source, mode):
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    # Keep encoding declarations, the module docstring and future imports in place.
    at = next(node.lineno - 1 for node in tree.body if not isinstance(node, ast.Expr)
              and not (isinstance(node, ast.ImportFrom) and node.module == "__future__"))
    changes = []
    if mode == "client":
        robot, = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Robot"]
        submit, = [node for node in robot.body if isinstance(node, ast.FunctionDef) and node.name == "submit"]
        if "note" not in [a.arg for a in (*submit.args.args, *submit.args.kwonlyargs)]:
            raise ValueError("Client lacks a native note parameter")
        first = submit.body[1] if ast.get_docstring(submit) is not None else submit.body[0]
        changes.append((first.lineno - 1, " " * first.col_offset + "arc_capture.require_note(note)\n"))
    else:
        method = mode == "service"
        if method:
            service, = [n for n in tree.body if isinstance(n, ast.ClassDef) and
                        any(isinstance(x, ast.FunctionDef) and x.name == "_submit" for x in n.body)]
            nodes = service.body
        else:
            nodes = tree.body
        for kind in ("submit", "observe"):
            name = ("_" if method else "do_") + kind
            fn, = [node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name]
            expected = (["self", "meta", "parts"] if method else
                        ["meta", "A"] if kind == "submit" else ["meta"])
            if [arg.arg for arg in fn.args.args] != expected:
                raise ValueError(f"Unexpected EFE signature for {name}")
            options = ", method=True, bad_request=bad_request" if method else ""
            changes.append((fn.lineno - 1, " " * fn.col_offset + f'@arc_capture.capture("{kind}"{options})\n'))
    for index, text in sorted(changes, reverse=True):
        lines.insert(index, text)
    lines.insert(at, "import arc_capture\n")
    result = "".join(lines)
    compile(result, "<efe>", "exec")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["client", "service", "server"])
    p.add_argument("source", type=Path)
    args = p.parse_args()
    args.source.write_text(instrument(args.source.read_text(), args.mode))

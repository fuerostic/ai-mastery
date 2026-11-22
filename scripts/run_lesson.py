"""Run a student solution and its tests. Prints a JSON object to stdout."""

import io
import json
import pathlib
import sys
import traceback

student = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
tests = pathlib.Path(sys.argv[2]).read_text(encoding="utf-8")
captured = io.StringIO()
previous = sys.stdout
sys.stdout = captured
namespace: dict = {"__name__": "__main__"}
ok = True
error = ""
trace = ""
try:
    exec(student, namespace, namespace)
    exec(tests, namespace, namespace)
except Exception as exc:
    ok = False
    error = f"{type(exc).__name__}: {exc}"
    trace = traceback.format_exc()
finally:
    sys.stdout = previous

sys.stdout.write(
    json.dumps(
        {
            "ok": ok,
            "error": error,
            "trace": trace[-12000:],
            "stdout": captured.getvalue()[:20000],
        }
    )
)

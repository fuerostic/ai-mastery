import { readFileSync } from "node:fs";
import { loadPyodide } from "pyodide";

const lessons = JSON.parse(readFileSync(process.argv[2], "utf8"));
const pyodide = await loadPyodide();
await pyodide.loadPackage("numpy");

const harness = `
import io, json, sys, traceback
_stdout = io.StringIO()
_previous = sys.stdout
sys.stdout = _stdout
_ns = {"__name__": "__main__"}
_ok = True
_error = ""
_trace = ""
try:
    exec(STUDENT_CODE, _ns, _ns)
    exec(TEST_CODE, _ns, _ns)
except Exception as exc:
    _ok = False
    _error = type(exc).__name__ + ": " + str(exc)
    _trace = traceback.format_exc()
finally:
    sys.stdout = _previous
json.dumps({
    "ok": _ok,
    "error": _error,
    "trace": _trace[-12000:],
    "stdout": _stdout.getvalue()[:20000],
})
`;

let failed = 0;
for (const lesson of lessons) {
  pyodide.globals.set("STUDENT_CODE", lesson.solution);
  pyodide.globals.set("TEST_CODE", lesson.tests);
  const raw = pyodide.runPython(harness);
  const result = JSON.parse(typeof raw === "string" ? raw : String(raw));
  if (!result.ok) {
    failed += 1;
    console.error(`\nPYODIDE FAIL ${lesson.id}: ${result.error}`);
    console.error(result.trace);
  } else {
    console.log(`pyodide pass ${lesson.id}`);
  }
}

if (failed) {
  console.error(`${failed} solution(s) failed in Pyodide`);
  process.exit(1);
}
console.log(`Pyodide: ${lessons.length} reference solutions passed.`);

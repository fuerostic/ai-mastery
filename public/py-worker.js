/* In-browser Python for coding exercises.
   Pyodide 0.29.5 — keep this version aligned with scripts/check-pyodide.mjs. */
const INDEX_URL = "https://cdn.jsdelivr.net/pyodide/v0.29.5/full/";

let loading = null;

function loadRuntime() {
  if (!loading) {
    loading = (async () => {
      importScripts(INDEX_URL + "pyodide.js");
      const pyodide = await loadPyodide({ indexURL: INDEX_URL });
      await pyodide.loadPackage("numpy");
      return pyodide;
    })();
  }
  return loading;
}

const HARNESS = `
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

self.onmessage = async (event) => {
  const data = event.data || {};
  const id = data.id;
  try {
    const pyodide = await loadRuntime();
    pyodide.globals.set("STUDENT_CODE", String(data.code ?? ""));
    pyodide.globals.set("TEST_CODE", String(data.tests ?? ""));
    const raw = pyodide.runPython(HARNESS);
    const text = typeof raw === "string" ? raw : String(raw);
    const payload = JSON.parse(text);
    self.postMessage({ id, runtimeReady: true, ...payload });
  } catch (err) {
    loading = null;
    self.postMessage({
      id,
      runtimeReady: false,
      ok: false,
      stdout: "",
      error: err && err.message ? String(err.message) : String(err),
      trace: "",
    });
  }
};

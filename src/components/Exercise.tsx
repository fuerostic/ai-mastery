import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import type { Exercise as ExerciseData } from "../content/types";
import { loadProgress } from "../lib/progressStore";
import { useProgress } from "../lib/progress";
import { pythonRuntimeReady, runExercise, type RunResult } from "../lib/runner";
import { Markdown } from "./MarkdownView";

export function Exercise({ lessonId, exercise }: { lessonId: string; exercise: ExerciseData }) {
  const { saveDraft, recordRun, resetCode } = useProgress();
  const [code, setCode] = useState(() => loadProgress().exercises[lessonId]?.code ?? exercise.starter);
  const [showKey, setShowKey] = useState(false);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("");
  const [result, setResult] = useState<RunResult | null>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const saved = loadProgress().exercises[lessonId]?.code;
    setCode(saved ?? exercise.starter);
    setShowKey(false);
    setResult(null);
    setStatus("");
  }, [lessonId, exercise.starter]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const current = loadProgress().exercises[lessonId];
      const baseline = current?.code ?? exercise.starter;
      if (baseline === code) return;
      saveDraft(lessonId, code);
    }, 350);
    return () => window.clearTimeout(timer);
  }, [code, lessonId, exercise.starter, saveDraft]);

  async function run() {
    setRunning(true);
    setStatus(
      pythonRuntimeReady()
        ? "Running tests…"
        : "Loading Python and NumPy. The first run downloads about 10 MB.",
    );
    const outcome = await runExercise(code, exercise.tests);
    setResult(outcome);
    setStatus("");
    setRunning(false);
    recordRun(lessonId, code, outcome.ok);
    window.setTimeout(() => resultRef.current?.scrollIntoView({ block: "nearest" }), 30);
  }

  function reset() {
    if (code !== exercise.starter && !window.confirm("Replace your code with the starter?")) return;
    setCode(exercise.starter);
    setResult(null);
    resetCode(lessonId, exercise.starter);
  }

  function loadKey() {
    if (!window.confirm("Replace the editor contents with the implementation key?")) return;
    setCode(exercise.solution);
    setResult(null);
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Tab") {
      event.preventDefault();
      const el = event.currentTarget;
      const start = el.selectionStart;
      const end = el.selectionEnd;
      const next = `${code.slice(0, start)}    ${code.slice(end)}`;
      setCode(next);
      window.requestAnimationFrame(() => {
        el.selectionStart = el.selectionEnd = start + 4;
      });
    }
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      if (!running) void run();
    }
  }

  return (
    <div className="exercise">
      <Markdown source={exercise.prompt} />
      <div className="editor">
        <div className="editor-bar">
          <span>exercise.py</span>
          <span>Ctrl or ⌘ Enter runs the tests</span>
        </div>
        <textarea
          value={code}
          onChange={(event) => setCode(event.target.value)}
          onKeyDown={onKeyDown}
          spellCheck={false}
          aria-label="Python exercise editor"
        />
      </div>
      <div className="exercise-actions">
        <button className="btn" type="button" onClick={() => void run()} disabled={running}>
          {running ? "Running…" : "Run tests"}
        </button>
        <button className="btn secondary" type="button" onClick={reset} disabled={running}>
          Reset starter
        </button>
        <button
          className="btn secondary"
          type="button"
          aria-expanded={showKey}
          onClick={() => setShowKey((value) => !value)}
        >
          {showKey ? "Hide implementation key" : "Show implementation key"}
        </button>
      </div>
      <p className="exercise-note">
        Tests stay off the page. A failing assertion tells you what the function should have done. Each run is a fresh
        program. Reload the page if an experiment patches NumPy itself.
      </p>
      {showKey ? (
        <div className="key-panel">
          <div className="editor-bar">
            <span>implementation_key.py</span>
            <button className="text-btn" type="button" onClick={loadKey}>
              Load into editor
            </button>
          </div>
          <pre>
            <code>{exercise.solution}</code>
          </pre>
        </div>
      ) : null}
      <div
        className={`console ${result ? (result.ok ? "pass" : "fail") : ""}`}
        aria-live="polite"
        ref={resultRef}
      >
        {status ? <p>{status}</p> : null}
        {result?.ok ? <p className="console-title">All tests passed.</p> : null}
        {result && !result.ok ? <p className="console-title">{result.error}</p> : null}
        {result?.stdout ? <pre>{result.stdout}</pre> : null}
        {result && !result.ok && result.trace ? (
          <details>
            <summary>Traceback</summary>
            <pre>{result.trace}</pre>
          </details>
        ) : null}
        {!status && !result ? <p className="console-quiet">Test output will land here.</p> : null}
      </div>
    </div>
  );
}

import { useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Heatmap } from "../components/Heatmap";
import { lessons, modules } from "../content/curriculum";
import { useProgress } from "../lib/progress";
import {
  currentStreak,
  lessonFraction,
  quizCorrectCount,
  sanitizeProgress,
} from "../lib/progressStore";
import { useTitle } from "../lib/useTitle";

export function DashboardPage() {
  useTitle("Progress");
  const { data, replaceAll, reset } = useProgress();
  const fileRef = useRef<HTMLInputElement>(null);
  const [message, setMessage] = useState("");

  const overall = lessons.reduce((sum, lesson) => sum + lessonFraction(lesson, data), 0) / lessons.length;
  const streak = currentStreak(data.activity);
  const passed = lessons.filter((lesson) => data.exercises[lesson.id]?.passed).length;
  const quizzes = lessons.filter((lesson) => quizCorrectCount(lesson, data) === lesson.quiz.length).length;
  const marked = lessons.filter((lesson) => data.completed[lesson.id]).length;
  const allMarked = marked === lessons.length;

  function exportProgress() {
    const payload = {
      version: 1,
      exportedAt: new Date().toISOString(),
      course: "ai-mastery",
      progress: data,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "ai-mastery-progress.json";
    link.click();
    URL.revokeObjectURL(url);
    setMessage("Exported ai-mastery-progress.json.");
  }

  async function importProgress(file: File) {
    try {
      const parsed = sanitizeProgress(JSON.parse(await file.text()));
      replaceAll(parsed);
      setMessage("Imported progress into this browser.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not read that file.");
    }
  }

  function erase() {
    if (!window.confirm("Erase all progress stored in this browser?")) return;
    reset();
    setMessage("Progress cleared on this device.");
  }

  return (
    <div className="page dashboard">
      <p className="eyebrow">Lab notebook</p>
      <h1>What you have actually done</h1>
      <p className="lede">
        A lesson counts as a third for a passing exercise, a third for the quiz, and a third for the completion mark.
        The mark is yours to give. The other two are checked.
      </p>

      <div className="dash-grid">
        <div className="stat-card ring-card">
          <Ring value={overall} />
          <div>
            <h2>Course</h2>
            <p>{Math.round(overall * 100)}% across {lessons.length} lessons.</p>
          </div>
        </div>
        <div className="stat-card">
          <p className="stat-num">{streak}</p>
          <h2>Day streak</h2>
          <p>Consecutive days with a run, a quiz answer, or a completion. Today still counts if you have not started it, as long as yesterday did.</p>
        </div>
        <div className="stat-card">
          <p className="stat-num">
            {passed}
            <small>/{lessons.length}</small>
          </p>
          <h2>Exercises passed</h2>
          <p>
            {quizzes} {quizzes === 1 ? "quiz" : "quizzes"} fully correct · {marked}{" "}
            {marked === 1 ? "lesson" : "lessons"} marked complete.
          </p>
        </div>
      </div>

      <section className="panel">
        <h2>Activity</h2>
        <Heatmap activity={data.activity} />
      </section>

      <section>
        <h2>By module</h2>
        <ul className="module-bars">
          {modules.map((module) => {
            const fraction =
              module.lessons.reduce((sum, lesson) => sum + lessonFraction(lesson, data), 0) /
              module.lessons.length;
            return (
              <li key={module.id}>
                <div>
                  <Link to={`/module/${module.id}`}>
                    0{module.number} {module.title}
                  </Link>
                  <span>{Math.round(fraction * 100)}%</span>
                </div>
                <span className="bar" aria-hidden="true">
                  <span style={{ width: `${Math.round(fraction * 100)}%` }} />
                </span>
              </li>
            );
          })}
        </ul>
      </section>

      <section className="panel portability">
        <h2>Your data</h2>
        <p>Progress never leaves this browser unless you export it. Import replaces what is stored here.</p>
        <div className="exercise-actions">
          <button className="btn" type="button" onClick={exportProgress}>
            Export JSON
          </button>
          <button className="btn secondary" type="button" onClick={() => fileRef.current?.click()}>
            Import JSON
          </button>
          <button className="btn secondary" type="button" onClick={erase}>
            Reset this browser
          </button>
          <Link className="btn secondary" to="/certificate">
            {allMarked ? "Open certificate" : "Certificate"}
          </Link>
        </div>
        <input
          ref={fileRef}
          className="file-input"
          type="file"
          accept="application/json,.json"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) void importProgress(file);
            event.target.value = "";
          }}
        />
        {message ? <p className="form-message">{message}</p> : null}
      </section>
    </div>
  );
}

function Ring({ value }: { value: number }) {
  const radius = 52;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - Math.min(1, Math.max(0, value)));
  return (
    <svg className="ring" viewBox="0 0 140 140" role="img" aria-label={`${Math.round(value * 100)} percent complete`}>
      <circle className="ring-track" cx="70" cy="70" r={radius} />
      <circle
        className="ring-value"
        cx="70"
        cy="70"
        r={radius}
        strokeDasharray={circumference}
        strokeDashoffset={offset}
      />
      <text x="70" y="76" textAnchor="middle">
        {Math.round(value * 100)}%
      </text>
    </svg>
  );
}

import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { lessons } from "../src/content/curriculum";
import { currentStreak, emptyProgress, lessonFraction, sanitizeProgress } from "../src/lib/progressStore";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

interface RunJson {
  ok: boolean;
  error: string;
  trace: string;
  stdout: string;
}

function runPython(student: string, tests: string): RunJson {
  const dir = mkdtempSync(path.join(tmpdir(), "aim-"));
  const studentPath = path.join(dir, "student.py");
  const testsPath = path.join(dir, "tests.py");
  writeFileSync(studentPath, student);
  writeFileSync(testsPath, tests);
  const result = spawnSync("python3", [path.join(root, "scripts/run_lesson.py"), studentPath, testsPath], {
    encoding: "utf8",
    timeout: 30_000,
  });
  rmSync(dir, { recursive: true, force: true });
  if (result.status === null) {
    return { ok: false, error: result.error?.message || "python timed out", trace: result.stderr ?? "", stdout: "" };
  }
  try {
    return JSON.parse(result.stdout) as RunJson;
  } catch {
    return {
      ok: false,
      error: "exercise runner returned invalid JSON",
      trace: `${result.stdout}\n${result.stderr}`,
      stdout: "",
    };
  }
}

function assert(condition: unknown, message: string): void {
  if (!condition) throw new Error(message);
}

assert(lessons.length >= 24, `expected at least 24 lessons, found ${lessons.length}`);

let failures = 0;
for (const lesson of lessons) {
  assert(lesson.quiz.length >= 3, `${lesson.id} needs 3 quiz questions`);
  assert(lesson.resources.length >= 3, `${lesson.id} needs resources`);
  const solution = runPython(lesson.exercise.solution, lesson.exercise.tests);
  if (!solution.ok) {
    failures += 1;
    console.error(`\nFAIL ${lesson.id}: ${solution.error}`);
    console.error(solution.trace);
  } else {
    console.log(`python pass ${lesson.id}`);
  }
  const starter = runPython(lesson.exercise.starter, lesson.exercise.tests);
  if (starter.ok) {
    failures += 1;
    console.error(`\nSTARTER unexpectedly passed ${lesson.id}`);
  }
}

const today = new Date(2026, 10, 22);
assert(currentStreak({ "2026-11-22": 1, "2026-11-21": 2, "2026-11-19": 4 }, today) === 2, "streak should stop at a gap");
assert(currentStreak({ "2026-11-21": 1 }, today) === 1, "yesterday should keep the streak alive");
assert(currentStreak({}, today) === 0, "empty activity has no streak");

const sample = lessons[0];
const blank = emptyProgress();
assert(lessonFraction(sample, blank) === 0, "untouched lesson scores zero");
const partial = {
  ...blank,
  quiz: { [sample.id]: { [sample.quiz[0].id]: sample.quiz[0].answer } },
};
const expected = 1 / sample.quiz.length / 3;
assert(Math.abs(lessonFraction(sample, partial) - expected) < 1e-9, "quiz credit should be one third of the quiz fraction");

const imported = sanitizeProgress({
  version: 1,
  exportedAt: "2026-11-22T00:00:00Z",
  progress: {
    version: 1,
    completed: { vectors: "2026-11-22T00:00:00Z" },
    quiz: { vectors: { q1: 1, q2: "nope" } },
    exercises: { vectors: { code: "print(1)", passed: true, updatedAt: "2026-11-22" } },
    activity: { "2026-11-22": 2, bad: 3 },
    learnerName: "Ada",
  },
});
assert(imported.learnerName === "Ada", "name should import");
assert(imported.completed.vectors.startsWith("2026"), "completion should import");
assert(imported.quiz.vectors.q1 === 1, "numeric quiz answers should import");
assert(imported.quiz.vectors.q2 === undefined, "non-numeric quiz answers should be dropped");
assert(imported.exercises.vectors.passed === true, "exercise pass flag should import");
assert(imported.activity["2026-11-22"] === 2, "activity dates should import");

let rejected = false;
try {
  sanitizeProgress({ version: 2 });
} catch {
  rejected = true;
}
assert(rejected, "version 2 files should be rejected");

if (failures) {
  console.error(`${failures} exercise check(s) failed`);
  process.exit(1);
}

const payloadPath = path.join(tmpdir(), "ai-mastery-exercises.json");
writeFileSync(
  payloadPath,
  JSON.stringify(lessons.map((lesson) => ({ id: lesson.id, solution: lesson.exercise.solution, tests: lesson.exercise.tests }))),
);
const pyodide = spawnSync("node", [path.join(root, "scripts/check-pyodide.mjs"), payloadPath], {
  stdio: "inherit",
  timeout: 300_000,
});
if (pyodide.status !== 0) process.exit(pyodide.status ?? 1);
console.log(`Checked ${lessons.length} lessons.`);

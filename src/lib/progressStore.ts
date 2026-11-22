import type { Lesson } from "../content/types";

export interface ExerciseState {
  code: string;
  passed: boolean;
  updatedAt: string;
}

export interface ProgressData {
  version: 1;
  completed: Record<string, string>;
  quiz: Record<string, Record<string, number>>;
  exercises: Record<string, ExerciseState>;
  activity: Record<string, number>;
  learnerName: string;
}

export const PROGRESS_KEY = "ai-mastery-progress-v1";

export function emptyProgress(): ProgressData {
  return {
    version: 1,
    completed: {},
    quiz: {},
    exercises: {},
    activity: {},
    learnerName: "",
  };
}

export function localISODate(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

export function addDays(date: Date, days: number): Date {
  const next = new Date(date);
  next.setDate(next.getDate() + days);
  return next;
}

export function startOfDay(date: Date): Date {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate());
}

function bump(data: ProgressData, now = new Date()): ProgressData {
  const key = localISODate(now);
  return {
    ...data,
    activity: { ...data.activity, [key]: (data.activity[key] ?? 0) + 1 },
  };
}

export function currentStreak(activity: Record<string, number>, today = new Date()): number {
  const active = new Set(Object.entries(activity).filter(([, count]) => count > 0).map(([day]) => day));
  let cursor = startOfDay(today);
  if (!active.has(localISODate(cursor))) {
    cursor = addDays(cursor, -1);
    if (!active.has(localISODate(cursor))) return 0;
  }
  let streak = 0;
  while (active.has(localISODate(cursor))) {
    streak += 1;
    cursor = addDays(cursor, -1);
  }
  return streak;
}

export function lessonFraction(lesson: Pick<Lesson, "id" | "quiz">, data: ProgressData): number {
  const answers = data.quiz[lesson.id] ?? {};
  const quizScore = lesson.quiz.length
    ? lesson.quiz.filter((question) => answers[question.id] === question.answer).length / lesson.quiz.length
    : 0;
  const exerciseScore = data.exercises[lesson.id]?.passed ? 1 : 0;
  const completeScore = data.completed[lesson.id] ? 1 : 0;
  return (quizScore + exerciseScore + completeScore) / 3;
}

export function quizCorrectCount(lesson: Pick<Lesson, "id" | "quiz">, data: ProgressData): number {
  const answers = data.quiz[lesson.id] ?? {};
  return lesson.quiz.filter((question) => answers[question.id] === question.answer).length;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function copyStringRecord(value: unknown): Record<string, string> {
  if (!isRecord(value)) return {};
  const out: Record<string, string> = {};
  for (const [key, entry] of Object.entries(value)) {
    if (key === "__proto__" || key === "constructor") continue;
    if (typeof entry === "string" && entry.length < 80) out[key] = entry;
  }
  return out;
}

export function sanitizeProgress(input: unknown): ProgressData {
  if (!isRecord(input)) throw new Error("That file is not a JSON object.");
  const root = isRecord(input.progress) ? input.progress : input;
  if (root.version !== 1) throw new Error("This file is not an AI Mastery progress export (version 1).");
  const completed = copyStringRecord(root.completed);
  const quiz: ProgressData["quiz"] = {};
  if (isRecord(root.quiz)) {
    for (const [lessonId, answers] of Object.entries(root.quiz)) {
      if (lessonId === "__proto__" || !isRecord(answers)) continue;
      const clean: Record<string, number> = {};
      for (const [questionId, choice] of Object.entries(answers)) {
        if (questionId === "__proto__") continue;
        if (typeof choice === "number" && Number.isInteger(choice) && choice >= 0 && choice < 12) {
          clean[questionId] = choice;
        }
      }
      quiz[lessonId] = clean;
    }
  }
  const exercises: ProgressData["exercises"] = {};
  if (isRecord(root.exercises)) {
    for (const [lessonId, state] of Object.entries(root.exercises)) {
      if (lessonId === "__proto__" || !isRecord(state)) continue;
      if (typeof state.code !== "string" || state.code.length > 200_000) continue;
      exercises[lessonId] = {
        code: state.code,
        passed: state.passed === true,
        updatedAt: typeof state.updatedAt === "string" ? state.updatedAt.slice(0, 40) : "",
      };
    }
  }
  const activity: Record<string, number> = {};
  if (isRecord(root.activity)) {
    for (const [day, count] of Object.entries(root.activity)) {
      if (!/^\d{4}-\d{2}-\d{2}$/.test(day)) continue;
      if (typeof count !== "number" || !Number.isFinite(count) || count <= 0) continue;
      activity[day] = Math.min(500, Math.round(count));
    }
  }
  const learnerName = typeof root.learnerName === "string" ? root.learnerName.slice(0, 80) : "";
  return { version: 1, completed, quiz, exercises, activity, learnerName };
}

export function loadProgress(): ProgressData {
  if (typeof localStorage === "undefined") return emptyProgress();
  try {
    const raw = localStorage.getItem(PROGRESS_KEY);
    if (!raw) return emptyProgress();
    return sanitizeProgress(JSON.parse(raw));
  } catch {
    return emptyProgress();
  }
}

export function saveProgress(data: ProgressData): void {
  if (typeof localStorage === "undefined") return;
  localStorage.setItem(PROGRESS_KEY, JSON.stringify(data));
}

export function withActivity(data: ProgressData, now = new Date()): ProgressData {
  return bump(data, now);
}

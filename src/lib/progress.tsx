import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import {
  emptyProgress,
  loadProgress,
  saveProgress,
  withActivity,
  type ProgressData,
} from "./progressStore";

interface ProgressApi {
  data: ProgressData;
  markComplete: (lessonId: string, done: boolean) => void;
  answerQuiz: (lessonId: string, questionId: string, choice: number) => void;
  saveDraft: (lessonId: string, code: string) => void;
  recordRun: (lessonId: string, code: string, passed: boolean) => void;
  resetCode: (lessonId: string, starter: string) => void;
  setLearnerName: (name: string) => void;
  replaceAll: (data: ProgressData) => void;
  reset: () => void;
}

const ProgressContext = createContext<ProgressApi | null>(null);

export function ProgressProvider({ children }: { children: ReactNode }) {
  const [data, setData] = useState<ProgressData>(() => loadProgress());

  const mutate = useCallback((recipe: (current: ProgressData) => ProgressData) => {
    setData((current) => {
      const next = recipe(current);
      saveProgress(next);
      return next;
    });
  }, []);

  const api = useMemo<ProgressApi>(
    () => ({
      data,
      markComplete: (lessonId, done) => {
        mutate((current) => {
          const completed = { ...current.completed };
          if (done) completed[lessonId] = new Date().toISOString();
          else delete completed[lessonId];
          const next = { ...current, completed };
          return done ? withActivity(next) : next;
        });
      },
      answerQuiz: (lessonId, questionId, choice) => {
        mutate((current) => {
          const lessonAnswers = { ...(current.quiz[lessonId] ?? {}), [questionId]: choice };
          return withActivity({
            ...current,
            quiz: { ...current.quiz, [lessonId]: lessonAnswers },
          });
        });
      },
      saveDraft: (lessonId, code) => {
        mutate((current) => {
          const previous = current.exercises[lessonId];
          if (previous?.code === code) return current;
          return {
            ...current,
            exercises: {
              ...current.exercises,
              [lessonId]: {
                code,
                passed: false,
                updatedAt: new Date().toISOString(),
              },
            },
          };
        });
      },
      recordRun: (lessonId, code, passed) => {
        mutate((current) =>
          withActivity({
            ...current,
            exercises: {
              ...current.exercises,
              [lessonId]: { code, passed, updatedAt: new Date().toISOString() },
            },
          }),
        );
      },
      resetCode: (lessonId, starter) => {
        mutate((current) => ({
          ...current,
          exercises: {
            ...current.exercises,
            [lessonId]: { code: starter, passed: false, updatedAt: new Date().toISOString() },
          },
        }));
      },
      setLearnerName: (name) => {
        mutate((current) => ({ ...current, learnerName: name.slice(0, 80) }));
      },
      replaceAll: (next) => {
        saveProgress(next);
        setData(next);
      },
      reset: () => {
        const blank = emptyProgress();
        saveProgress(blank);
        setData(blank);
      },
    }),
    [data, mutate],
  );

  return <ProgressContext.Provider value={api}>{children}</ProgressContext.Provider>;
}

export function useProgress(): ProgressApi {
  const api = useContext(ProgressContext);
  if (!api) throw new Error("useProgress must be used inside ProgressProvider");
  return api;
}

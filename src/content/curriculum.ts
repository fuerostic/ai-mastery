import { modulesMeta } from "./modules";
import { parseLesson } from "./parseLesson";
import type { Lesson, Module } from "./types";

const rawLessons = import.meta.glob("./lessons/*.md", {
  eager: true,
  query: "?raw",
  import: "default",
}) as Record<string, string>;

const byId = new Map<string, Lesson>();

for (const [path, source] of Object.entries(rawLessons)) {
  if (typeof source !== "string") {
    throw new Error(`${path} did not import as raw text`);
  }
  const lesson = parseLesson(source, path);
  if (byId.has(lesson.id)) throw new Error(`Duplicate lesson id ${lesson.id}`);
  byId.set(lesson.id, lesson);
}

export const modules: Module[] = modulesMeta.map((meta) => {
  const lessons = meta.lessonIds.map((id) => {
    const lesson = byId.get(id);
    if (!lesson) throw new Error(`No lesson file for id "${id}"`);
    if (lesson.module !== meta.id) {
      throw new Error(`Lesson ${id} is filed under ${lesson.module}, listed in ${meta.id}`);
    }
    return lesson;
  });
  return {
    id: meta.id,
    number: meta.number,
    title: meta.title,
    tagline: meta.tagline,
    description: meta.description,
    lessons,
  };
});

for (const id of byId.keys()) {
  if (!modules.some((module) => module.lessons.some((lesson) => lesson.id === id))) {
    throw new Error(`Lesson ${id} is not listed in src/content/modules.ts`);
  }
}

export const lessons: Lesson[] = modules.flatMap((module) => module.lessons);

export function moduleById(id: string): Module | undefined {
  return modules.find((module) => module.id === id);
}

export function lessonById(id: string): Lesson | undefined {
  return lessons.find((lesson) => lesson.id === id);
}

export function moduleOf(lessonId: string): Module | undefined {
  return modules.find((module) => module.lessons.some((lesson) => lesson.id === lessonId));
}

export function lessonNeighbors(lessonId: string): {
  prev?: { lesson: Lesson; module: Module };
  next?: { lesson: Lesson; module: Module };
} {
  const index = lessons.findIndex((lesson) => lesson.id === lessonId);
  if (index < 0) return {};
  const prevLesson = lessons[index - 1];
  const nextLesson = lessons[index + 1];
  return {
    prev: prevLesson ? { lesson: prevLesson, module: moduleOf(prevLesson.id)! } : undefined,
    next: nextLesson ? { lesson: nextLesson, module: moduleOf(nextLesson.id)! } : undefined,
  };
}

export interface SearchHit {
  lesson: Lesson;
  module: Module;
  titleScore: number;
}

export function searchLessons(query: string): SearchHit[] {
  const terms = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
  if (terms.length === 0) return [];
  const hits: SearchHit[] = [];
  for (const module of modules) {
    for (const lesson of module.lessons) {
      const title = lesson.title.toLowerCase();
      const haystack = `${title} ${module.title.toLowerCase()} ${lesson.searchText}`;
      if (!terms.every((term) => haystack.includes(term))) continue;
      const titleScore = terms.reduce((score, term) => score + (title.includes(term) ? 2 : 0), 0);
      hits.push({ lesson, module, titleScore });
    }
  }
  hits.sort((a, b) => b.titleScore - a.titleScore || a.module.number - b.module.number);
  return hits;
}

export function totalMinutes(): number {
  return lessons.reduce((sum, lesson) => sum + lesson.minutes, 0);
}

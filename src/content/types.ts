export type ResourceKind = "video" | "reading" | "course" | "book" | "paper" | "code";

export interface Resource {
  title: string;
  url: string;
  kind: ResourceKind;
  source: string;
  note: string;
}

export interface QuizQuestion {
  id: string;
  prompt: string;
  choices: string[];
  answer: number;
  explanation: string;
}

export interface Exercise {
  prompt: string;
  starter: string;
  solution: string;
  tests: string;
}

export interface Lesson {
  id: string;
  module: string;
  title: string;
  minutes: number;
  summary: string;
  idea: string;
  intuition: string;
  math: string;
  resources: Resource[];
  exercise: Exercise;
  quiz: QuizQuestion[];
  searchText: string;
}

export interface ModuleMeta {
  id: string;
  number: number;
  title: string;
  tagline: string;
  description: string;
  lessonIds: string[];
}

export interface Module extends Omit<ModuleMeta, "lessonIds"> {
  lessons: Lesson[];
}

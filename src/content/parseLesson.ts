import type { Lesson, QuizQuestion, Resource, ResourceKind } from "./types";

const KINDS = new Set<ResourceKind>(["video", "reading", "course", "book", "paper", "code"]);

function stripMarkdown(source: string): string {
  return source
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/\$\$[\s\S]*?\$\$/g, " ")
    .replace(/\$[^$\n]+\$/g, " ")
    .replace(/[#>*_`[\]]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
}

function frontMatter(text: string, filename: string): { meta: Record<string, string>; body: string } {
  if (!text.startsWith("---\n")) {
    throw new Error(`${filename}: front matter must start the file`);
  }
  const end = text.indexOf("\n---\n", 4);
  if (end < 0) throw new Error(`${filename}: unterminated front matter`);
  const meta: Record<string, string> = {};
  for (const line of text.slice(4, end).split("\n")) {
    if (!line.trim()) continue;
    const idx = line.indexOf(":");
    if (idx <= 0) throw new Error(`${filename}: bad front matter line "${line}"`);
    meta[line.slice(0, idx).trim()] = line.slice(idx + 1).trim();
  }
  return { meta, body: text.slice(end + 5) };
}

function sectionsOf(body: string, filename: string): Map<string, string> {
  const matches = [...body.matchAll(/^## (?!#)(.+)$/gm)];
  if (matches.length === 0) throw new Error(`${filename}: no sections`);
  const sections = new Map<string, string>();
  matches.forEach((match, index) => {
    const name = match[1].trim();
    const start = (match.index ?? 0) + match[0].length;
    const end = index + 1 < matches.length ? (matches[index + 1].index ?? body.length) : body.length;
    sections.set(name, body.slice(start, end).trim());
  });
  return sections;
}

function requireSection(sections: Map<string, string>, name: string, filename: string): string {
  const value = sections.get(name);
  if (!value) throw new Error(`${filename}: missing ## ${name}`);
  return value;
}

function parseResources(section: string, filename: string): Resource[] {
  const resources: Resource[] = [];
  for (const line of section.split("\n")) {
    if (!line.trim()) continue;
    const match = line.match(
      /^- \[(.+?)\]\((https?:\/\/[^)\s]+)\) · (video|reading|course|book|paper|code) · ([^·]+) · (.+)$/,
    );
    if (!match) throw new Error(`${filename}: bad resource line "${line}"`);
    const kind = match[3] as ResourceKind;
    if (!KINDS.has(kind)) throw new Error(`${filename}: bad resource kind ${kind}`);
    resources.push({
      title: match[1].trim(),
      url: match[2].trim(),
      kind,
      source: match[4].trim(),
      note: match[5].trim(),
    });
  }
  if (resources.length < 3) throw new Error(`${filename}: need at least 3 resources`);
  return resources;
}

function parseQuiz(section: string, filename: string): QuizQuestion[] {
  const chunks = section
    .split(/\n(?=### )/)
    .map((chunk) => chunk.trim())
    .filter(Boolean);
  const questions: QuizQuestion[] = [];
  for (const chunk of chunks) {
    if (!chunk.startsWith("### ")) {
      throw new Error(`${filename}: quiz text before the first question is not allowed`);
    }
    const newline = chunk.indexOf("\n");
    if (newline < 0) throw new Error(`${filename}: empty quiz question`);
    const id = chunk.slice(4, newline).trim();
    const lines = chunk.slice(newline + 1).split("\n");
    const prompt: string[] = [];
    const choices: { text: string; correct: boolean }[] = [];
    const explanation: string[] = [];
    let mode: "prompt" | "choices" | "explanation" = "prompt";
    for (const line of lines) {
      if (mode === "prompt" && /^- \[[ xX]\] /.test(line)) mode = "choices";
      if (mode === "choices" && line.startsWith(">")) mode = "explanation";
      if (mode === "prompt") {
        prompt.push(line);
        continue;
      }
      if (mode === "choices") {
        if (!line.trim()) continue;
        const match = line.match(/^- \[([ xX])\] (.*)$/);
        if (!match) throw new Error(`${filename}: bad choice "${line}"`);
        choices.push({ text: match[2].trim(), correct: match[1].toLowerCase() === "x" });
        continue;
      }
      if (line.startsWith(">")) explanation.push(line.replace(/^>\s?/, ""));
      else if (line.trim()) throw new Error(`${filename}: explanation lines must start with >`);
    }
    const answer = choices.findIndex((choice) => choice.correct);
    const correctCount = choices.filter((choice) => choice.correct).length;
    if (!id || !prompt.join("").trim()) throw new Error(`${filename}: quiz question needs an id and a prompt`);
    if (choices.length < 3) throw new Error(`${filename}: ${id} needs at least 3 choices`);
    if (correctCount !== 1) throw new Error(`${filename}: ${id} needs exactly one correct choice`);
    if (!explanation.join("").trim()) throw new Error(`${filename}: ${id} needs an explanation`);
    questions.push({
      id,
      prompt: prompt.join("\n").trim(),
      choices: choices.map((choice) => choice.text),
      answer,
      explanation: explanation.join(" ").trim(),
    });
  }
  if (questions.length < 3) throw new Error(`${filename}: need at least 3 quiz questions`);
  return questions;
}

function fence(section: string, label: string, filename: string): string {
  const match = section.match(new RegExp("```python " + label + "\\n([\\s\\S]*?)\\n```"));
  if (!match) throw new Error(`${filename}: missing \`\`\`python ${label} fence`);
  return match[1].replace(/\s+$/, "");
}

function parseExercise(section: string, filename: string) {
  const starter = fence(section, "starter", filename);
  const solution = fence(section, "solution", filename);
  const tests = fence(section, "tests", filename);
  const promptEnd = section.search(/```python (?:starter|solution|tests)\n/);
  const prompt = (promptEnd < 0 ? section : section.slice(0, promptEnd)).trim();
  if (!prompt || !starter || !solution || !tests) {
    throw new Error(`${filename}: exercise prompt, starter, solution, and tests are all required`);
  }
  if (!starter.includes("NotImplementedError")) {
    throw new Error(`${filename}: starter should leave the work as NotImplementedError`);
  }
  if (solution.includes("NotImplementedError")) {
    throw new Error(`${filename}: solution still contains NotImplementedError`);
  }
  return { prompt, starter, solution, tests };
}

export function parseLesson(markdown: string, filename: string): Lesson {
  const text = markdown.replace(/^\uFEFF/, "").replace(/\r\n/g, "\n");
  const { meta, body } = frontMatter(text, filename);
  for (const key of ["id", "module", "title", "minutes", "summary", "idea"]) {
    if (!meta[key]) throw new Error(`${filename}: missing front matter field ${key}`);
  }
  const minutes = Number(meta.minutes);
  if (!Number.isInteger(minutes) || minutes <= 0) {
    throw new Error(`${filename}: minutes must be a positive integer`);
  }
  const sections = sectionsOf(body, filename);
  const intuition = requireSection(sections, "Intuition", filename);
  const math = requireSection(sections, "Math", filename);
  const resources = parseResources(requireSection(sections, "Resources", filename), filename);
  const exercise = parseExercise(requireSection(sections, "Exercise", filename), filename);
  const quiz = parseQuiz(requireSection(sections, "Quiz", filename), filename);
  const lesson: Lesson = {
    id: meta.id,
    module: meta.module,
    title: meta.title,
    minutes,
    summary: meta.summary,
    idea: meta.idea,
    intuition,
    math,
    resources,
    exercise,
    quiz,
    searchText: "",
  };
  lesson.searchText = [
    lesson.title,
    lesson.summary,
    lesson.idea,
    stripMarkdown(lesson.intuition),
    stripMarkdown(lesson.math),
    lesson.resources.map((resource) => `${resource.title} ${resource.source} ${resource.note}`).join(" "),
  ]
    .join(" ")
    .toLowerCase();
  return lesson;
}

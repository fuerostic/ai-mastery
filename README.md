# AI Mastery

A free, math-first self-study site for someone rebuilding fundamentals on the way to modern language models. It was written with an M.Sc. AI student at the University of Bonn in mind. It is not an official university course, and finishing it is not a credential.

Twenty-five lessons sit in six modules:

1. Math foundations for machine learning
2. NumPy and classical learning
3. Neural nets from scratch
4. Transformers
5. Training language models
6. Research skills for NLP

Each lesson has an intuition, the equations that carry it, links to free lectures and papers, a Python exercise that runs in the browser, three quiz questions, and a completion checkbox. Progress stays in `localStorage`. You can export and import it as JSON.

The prose is original to this repository. Videos, books, and papers are linked, not copied.

## Run it locally

You need Node.js 22 or newer.

```bash
npm install
npm run dev
```

Open the URL Vite prints. The site is served under `/ai-mastery/` so it matches GitHub Pages, typically:

```text
http://127.0.0.1:5173/ai-mastery/
```

The first time you press **Run tests**, the page downloads Pyodide and NumPy (about 10 MB) from the jsDelivr CDN. Later runs reuse that runtime until you reload.

```bash
npm run build
npm run preview
```

Preview uses the same `/ai-mastery/` base, on port 4173.

## Check the exercises

Reference solutions must pass their tests in CPython (with NumPy) and in Pyodide.

```bash
python3 -m pip install numpy
npm run check:exercises
```

`scripts/check-exercises.ts` loads every lesson, runs each solution, checks that the starter does not already pass, then repeats the solutions inside Pyodide 0.29.5. The browser worker in `public/py-worker.js` uses that same Pyodide version.

## Add a lesson

Lessons are Markdown files in `src/content/lessons/`. Copy an existing file. The front matter needs `id`, `module`, `title`, `minutes`, `summary`, and `idea`. The body needs these sections, in order: `## Intuition`, `## Math`, `## Resources`, `## Exercise`, `## Quiz`.

Resources are one line each:

```text
- [Title](https://example.com) · video · Source name · Why this link is here.
```

The kind is one of `video`, `reading`, `course`, `book`, `paper`, `code`.

The exercise section is a short prompt, then three fences labeled `python starter`, `python solution`, and `python tests`. The starter should raise `NotImplementedError`. Tests call the functions by name and are not shown in the UI.

Quiz questions look like this:

```text
### q1
Prompt, with $inline math$ if you need it.

- [ ] a wrong answer
- [x] the right answer
- [ ] another wrong answer

> Why the right answer is right.
```

Register the new id in `src/content/modules.ts` under the right module. Then run `npm run check:exercises` and `npm run dev`.

## Deploy on GitHub Pages

The workflow in `.github/workflows/pages.yml` builds the site and deploys `dist/` to GitHub Pages on every push to `main`.

1. Merge this branch to `main` (or push the workflow to `main`).
2. In the repository on GitHub, open **Settings → Pages**.
3. Under **Build and deployment**, set **Source** to **GitHub Actions**.
4. Push to `main`, or run the **Deploy GitHub Pages** workflow by hand.
5. When the deploy job finishes, the site is at:

```text
https://fuerostic.github.io/ai-mastery/
```

Deep links work because `public/404.html` sends unknown Pages paths back to the single-page app. Progress, quiz answers, and exercise code are still stored only in the visitor's browser.

## Progress file

**Export JSON** on the progress page writes a file like:

```json
{
  "version": 1,
  "exportedAt": "2026-11-22T00:00:00.000Z",
  "course": "ai-mastery",
  "progress": {
    "version": 1,
    "completed": { "vectors": "2026-11-22T00:00:00.000Z" },
    "quiz": { "vectors": { "q1": 1 } },
    "exercises": { "vectors": { "code": "...", "passed": true, "updatedAt": "..." } },
    "activity": { "2026-11-22": 3 },
    "learnerName": "Ada"
  }
}
```

Import replaces whatever this browser had stored. A lesson's displayed percent is one third for a passing exercise, one third for the fraction of quiz questions currently correct, and one third for the completion checkbox.

## Stack

Vite, React, React Router, KaTeX, and Pyodide. No backend. The Python runtime is loaded from the CDN the first time an exercise runs.

## License

The original writing and site code in this repository are released under the MIT license. Linked materials remain under their own licenses and authors.

import { Link } from "react-router-dom";
import { lessons, modules, totalMinutes } from "../content/curriculum";
import { useProgress } from "../lib/progress";
import { lessonFraction } from "../lib/progressStore";
import { useTitle } from "../lib/useTitle";

export function HomePage() {
  useTitle("AI Mastery");
  const { data } = useProgress();
  const hours = Math.round(totalMinutes() / 60);
  const next = lessons.find((lesson) => !data.completed[lesson.id]) ?? lessons[0];
  const anyProgress = lessons.some((lesson) => lessonFraction(lesson, data) > 0);

  return (
    <div className="page wide">
      <section className="hero">
        <p className="eyebrow">Independent self-study · math first</p>
        <h1>Rebuild the math. Then build the model.</h1>
        <p className="lede">
          A free path from linear algebra to modern language models, written for someone rebuilding fundamentals on the
          way through an M.Sc. in AI. Every lesson has the idea, the equations, a program you run in the browser, and
          three questions. Nothing here is locked behind a login.
        </p>
        <div className="hero-actions">
          <Link className="btn" to={next ? `/lesson/${next.id}` : "/dashboard"}>
            {anyProgress ? `Continue · ${next.title}` : "Start with vectors"}
          </Link>
          <Link className="btn secondary" to="/dashboard">
            Open the lab notebook
          </Link>
        </div>
        <dl className="stat-row">
          <div>
            <dt>Modules</dt>
            <dd>6</dd>
          </div>
          <div>
            <dt>Lessons</dt>
            <dd>{lessons.length}</dd>
          </div>
          <div>
            <dt>Guided study</dt>
            <dd>{hours} h</dd>
          </div>
          <div>
            <dt>Server</dt>
            <dd>None</dd>
          </div>
        </dl>
      </section>

      <section className="panel steps">
        <h2>How a lesson works</h2>
        <ol>
          <li>
            <strong>Read the mechanism.</strong>
            <span>Intuition first, then the two or three equations that carry the idea.</span>
          </li>
          <li>
            <strong>Leave the page when you should.</strong>
            <span>Each lesson points at free lectures, books, and papers. Those are the long version.</span>
          </li>
          <li>
            <strong>Implement it.</strong>
            <span>Python and NumPy run here. Hidden tests check your functions and show the assertion that failed.</span>
          </li>
          <li>
            <strong>Keep a record.</strong>
            <span>Quiz answers, code, and a completion mark stay in this browser. Export them if you want a backup.</span>
          </li>
        </ol>
      </section>

      <section>
        <div className="section-head">
          <h2>The path</h2>
          <p>About {hours} hours on this site. The linked courses are much longer, and that is the point.</p>
        </div>
        <ol className="module-grid">
          {modules.map((module) => {
            const fraction =
              module.lessons.reduce((sum, lesson) => sum + lessonFraction(lesson, data), 0) /
              module.lessons.length;
            const minutes = module.lessons.reduce((sum, lesson) => sum + lesson.minutes, 0);
            return (
              <li key={module.id}>
                <Link to={`/module/${module.id}`} className="module-card">
                  <span className="module-index">0{module.number}</span>
                  <h3>{module.title}</h3>
                  <p>{module.tagline}</p>
                  <div className="module-meta">
                    <span>{module.lessons.length} lessons</span>
                    <span>{minutes} min</span>
                    <span>{Math.round(fraction * 100)}%</span>
                  </div>
                  <span className="bar" aria-hidden="true">
                    <span style={{ width: `${Math.round(fraction * 100)}%` }} />
                  </span>
                </Link>
              </li>
            );
          })}
        </ol>
      </section>

      <section className="panel note">
        <h2>What this is for</h2>
        <p>
          University of Bonn students in the M.Sc. AI program often arrive with uneven math and programming, then meet
          transformers in the same year. This course is a place to close that gap in order: linear algebra and
          probability, classical models, networks from scratch, attention, training, and the habits that make an NLP
          number believable.
        </p>
        <p>
          It is not an official university offering, and finishing it is not a credential. The writing is original to
          this site. The videos and books belong to their authors.
        </p>
      </section>
      <footer className="site-foot">
        Progress lives in localStorage on this device. The Python runtime is loaded from the Pyodide CDN the first time
        you run an exercise.
      </footer>
    </div>
  );
}

import { Link, useParams } from "react-router-dom";
import { Exercise } from "../components/Exercise";
import { Markdown } from "../components/MarkdownView";
import { Quiz } from "../components/Quiz";
import { lessonById, lessonNeighbors, moduleOf } from "../content/curriculum";
import { useProgress } from "../lib/progress";
import { useTitle } from "../lib/useTitle";

const KIND_LABEL: Record<string, string> = {
  video: "Video",
  reading: "Reading",
  course: "Course",
  book: "Book",
  paper: "Paper",
  code: "Code",
};

export function LessonPage() {
  const { lessonId } = useParams();
  const lesson = lessonId ? lessonById(lessonId) : undefined;
  const module = lesson ? moduleOf(lesson.id) : undefined;
  useTitle(lesson ? lesson.title : "Lesson");
  const { data, markComplete } = useProgress();

  if (!lesson || !module) {
    return (
      <div className="page">
        <h1>That lesson is not in the curriculum.</h1>
        <Link to="/">Back to the start</Link>
      </div>
    );
  }

  const neighbors = lessonNeighbors(lesson.id);
  const index = module.lessons.findIndex((item) => item.id === lesson.id) + 1;
  const done = Boolean(data.completed[lesson.id]);

  return (
    <article className="page lesson">
      <header className="lesson-hero">
        <p className="eyebrow">
          <Link to={`/module/${module.id}`}>
            Module {module.number} · {module.title}
          </Link>
        </p>
        <h1>{lesson.title}</h1>
        <p className="summary">{lesson.summary}</p>
        <p className="lesson-meta">
          Lesson {index} of {module.lessons.length} · {lesson.minutes} minutes on this page
        </p>
        <blockquote className="idea">{lesson.idea}</blockquote>
        <nav className="jump" aria-label="On this lesson">
          <a href="#intuition">Intuition</a>
          <a href="#math">Math</a>
          <a href="#resources">Resources</a>
          <a href="#exercise">Exercise</a>
          <a href="#quiz">Quiz</a>
          <a href="#complete">Complete</a>
        </nav>
      </header>

      <section id="intuition" className="lesson-section">
        <h2>Intuition</h2>
        <Markdown source={lesson.intuition} />
      </section>

      <section id="math" className="lesson-section">
        <h2>The math</h2>
        <Markdown source={lesson.math} />
      </section>

      <section id="resources" className="lesson-section">
        <h2>Free resources</h2>
        <p className="section-intro">Leave this page for the long version. These are other people’s courses, books, and papers.</p>
        <ul className="resource-list">
          {lesson.resources.map((resource) => (
            <li key={resource.url}>
              <a href={resource.url} target="_blank" rel="noreferrer">
                <span className={`kind kind-${resource.kind}`}>{KIND_LABEL[resource.kind]}</span>
                <strong>{resource.title}</strong>
                <small>{resource.source}</small>
                <span>{resource.note}</span>
              </a>
            </li>
          ))}
        </ul>
      </section>

      <section id="exercise" className="lesson-section">
        <h2>Coding exercise</h2>
        <Exercise lessonId={lesson.id} exercise={lesson.exercise} />
      </section>

      <section id="quiz" className="lesson-section">
        <h2>Check yourself</h2>
        <Quiz lessonId={lesson.id} questions={lesson.quiz} />
      </section>

      <section id="complete" className="lesson-section">
        <label className={done ? "complete-card done" : "complete-card"}>
          <input
            type="checkbox"
            checked={done}
            onChange={(event) => markComplete(lesson.id, event.target.checked)}
          />
          <span>
            <strong>{done ? "Lesson marked complete" : "Mark this lesson complete"}</strong>
            <small>Your own attestation. It is stored only in this browser, and you can uncheck it.</small>
          </span>
        </label>
      </section>

      <nav className="neighbors" aria-label="Adjacent lessons">
        {neighbors.prev ? (
          <Link to={`/lesson/${neighbors.prev.lesson.id}`}>
            <small>Previous</small>
            {neighbors.prev.lesson.title}
          </Link>
        ) : (
          <span />
        )}
        {neighbors.next ? (
          <Link to={`/lesson/${neighbors.next.lesson.id}`} className="next">
            <small>Next</small>
            {neighbors.next.lesson.title}
          </Link>
        ) : (
          <Link to="/certificate" className="next">
            <small>Finish</small>
            Certificate
          </Link>
        )}
      </nav>
    </article>
  );
}

import { Link, useParams } from "react-router-dom";
import { moduleById } from "../content/curriculum";
import { useProgress } from "../lib/progress";
import { lessonFraction, quizCorrectCount } from "../lib/progressStore";
import { useTitle } from "../lib/useTitle";

export function ModulePage() {
  const { moduleId } = useParams();
  const module = moduleId ? moduleById(moduleId) : undefined;
  useTitle(module ? module.title : "Module");
  const { data } = useProgress();

  if (!module) {
    return (
      <div className="page">
        <h1>That module is not in the curriculum.</h1>
        <Link to="/">Back to the start</Link>
      </div>
    );
  }

  const fraction =
    module.lessons.reduce((sum, lesson) => sum + lessonFraction(lesson, data), 0) / module.lessons.length;

  return (
    <div className="page">
      <p className="eyebrow">Module {module.number}</p>
      <h1>{module.title}</h1>
      <p className="lede">{module.description}</p>
      <p className="module-progress">{Math.round(fraction * 100)}% of this module, averaged across exercise, quiz, and completion.</p>
      <ol className="lesson-list">
        {module.lessons.map((lesson, index) => {
          const score = lessonFraction(lesson, data);
          const done = Boolean(data.completed[lesson.id]);
          const passed = Boolean(data.exercises[lesson.id]?.passed);
          const quiz = quizCorrectCount(lesson, data);
          return (
            <li key={lesson.id}>
              <Link to={`/lesson/${lesson.id}`}>
                <span className="lesson-num">{index + 1}</span>
                <span>
                  <strong>{lesson.title}</strong>
                  <small>{lesson.summary}</small>
                  <span className="pills">
                    <em className={passed ? "on" : ""}>Exercise</em>
                    <em className={quiz === lesson.quiz.length ? "on" : ""}>
                      Quiz {quiz}/{lesson.quiz.length}
                    </em>
                    <em className={done ? "on" : ""}>Marked</em>
                    <em>{lesson.minutes} min</em>
                  </span>
                </span>
                <span className="lesson-pct">{Math.round(score * 100)}%</span>
              </Link>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

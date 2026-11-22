import { Link } from "react-router-dom";
import { lessons, modules } from "../content/curriculum";
import { useProgress } from "../lib/progress";
import { useTitle } from "../lib/useTitle";

export function CertificatePage() {
  useTitle("Certificate");
  const { data, setLearnerName } = useProgress();
  const marked = lessons.filter((lesson) => data.completed[lesson.id]);
  const ready = marked.length === lessons.length;
  const finishedAt = marked
    .map((lesson) => data.completed[lesson.id])
    .sort()
    .at(-1);
  const finishedLabel = finishedAt
    ? new Date(finishedAt).toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" })
    : "";

  return (
    <div className="page">
      <p className="eyebrow no-print">Certificate</p>
      <div className="no-print">
        <h1>{ready ? "You marked every lesson." : "The certificate waits on the completion marks."}</h1>
        <p className="lede">
          {ready
            ? "Print or save this page as a PDF. It records a self-attested finish, not an exam."
            : `${marked.length} of ${lessons.length} lessons are marked complete. Exercises and quizzes are tracked on the dashboard; this page follows the checkboxes.`}
        </p>
      </div>
      <article className={ready ? "certificate" : "certificate locked"}>
        <p className="cert-brand">AI Mastery</p>
        <h2>Certificate of completion</h2>
        <p className="cert-kicker">From foundations to frontier models</p>
        <label className="name-field no-print">
          Name on the certificate
          <input
            value={data.learnerName}
            maxLength={80}
            onChange={(event) => setLearnerName(event.target.value)}
            placeholder="Your name"
          />
        </label>
        <p className="cert-name">{data.learnerName.trim() || "Your name"}</p>
        <p className="cert-body">
          has worked through the lessons of this independent course: linear algebra and probability, classical models,
          neural nets from scratch, transformers, language-model training, and the measurement habits of NLP research.
        </p>
        <ul>
          {modules.map((module) => (
            <li key={module.id}>
              <span>0{module.number}</span>
              {module.title}
            </li>
          ))}
        </ul>
        <p className="cert-date">{ready ? `Marked complete ${finishedLabel}` : "Not yet complete"}</p>
        <p className="cert-disclaimer">
          Self-attested progress stored in one browser. This is not an official University of Bonn credential, it is
          not proctored, and it does not certify employment or admission.
        </p>
      </article>
      <div className="exercise-actions no-print">
        <button className="btn" type="button" disabled={!ready} onClick={() => window.print()}>
          Print or save PDF
        </button>
        <Link className="btn secondary" to="/dashboard">
          Back to progress
        </Link>
      </div>
    </div>
  );
}

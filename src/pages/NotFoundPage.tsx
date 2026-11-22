import { Link } from "react-router-dom";
import { useTitle } from "../lib/useTitle";

export function NotFoundPage() {
  useTitle("Not found");
  return (
    <div className="page">
      <p className="eyebrow">404</p>
      <h1>That page is not part of the course.</h1>
      <p className="lede">The curriculum is a fixed set of lessons. Search from the top bar, or go back to the start.</p>
      <Link className="btn" to="/">
        Back to the start
      </Link>
    </div>
  );
}

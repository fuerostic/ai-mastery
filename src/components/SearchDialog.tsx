import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { searchLessons } from "../content/curriculum";

export function SearchDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [query, setQuery] = useState("");
  const [cursor, setCursor] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const openedAt = useRef(0);
  const navigate = useNavigate();
  const hits = useMemo(() => searchLessons(query).slice(0, 12), [query]);

  useLayoutEffect(() => {
    if (!open) return;
    openedAt.current = performance.now();
    setCursor(0);
    inputRef.current?.focus();
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previous;
    };
  }, [open]);

  useEffect(() => setCursor(0), [query]);

  if (!open) return null;

  function go(index: number) {
    const hit = hits[index];
    if (!hit) return;
    navigate(`/lesson/${hit.lesson.id}`);
    onClose();
    setQuery("");
  }

  return (
    <div
        className="search-scrim"
        role="presentation"
        onMouseDown={() => {
          if (performance.now() - openedAt.current < 250) return;
          onClose();
        }}
      >
      <div
        className="search-dialog"
        role="dialog"
        aria-modal="true"
        aria-label="Search lessons"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <input
          ref={inputRef}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search the curriculum"
          aria-label="Search the curriculum"
          onKeyDown={(event) => {
            if (event.key === "Escape") onClose();
            if (event.key === "ArrowDown") {
              event.preventDefault();
              setCursor((value) => Math.min(value + 1, Math.max(hits.length - 1, 0)));
            }
            if (event.key === "ArrowUp") {
              event.preventDefault();
              setCursor((value) => Math.max(value - 1, 0));
            }
            if (event.key === "Enter") go(cursor);
          }}
        />
        {query.trim() && hits.length === 0 ? <p className="search-empty">No lesson matches that.</p> : null}
        <ul>
          {hits.map((hit, index) => (
            <li key={hit.lesson.id}>
              <button
                type="button"
                className={index === cursor ? "search-hit active" : "search-hit"}
                onMouseEnter={() => setCursor(index)}
                onClick={() => go(index)}
              >
                <span>{hit.lesson.title}</span>
                <small>
                  Module {hit.module.number} · {hit.module.title}
                </small>
              </button>
            </li>
          ))}
        </ul>
        {!query.trim() ? <p className="search-hint">Try “condition number”, “causal mask”, or “kappa”.</p> : null}
      </div>
    </div>
  );
}

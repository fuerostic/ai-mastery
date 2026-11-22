import { useEffect, useState } from "react";
import { Link, Outlet, useLocation } from "react-router-dom";
import { lessons, modules } from "../content/curriculum";
import { useProgress } from "../lib/progress";
import { lessonFraction } from "../lib/progressStore";
import { SearchDialog } from "./SearchDialog";

function Mark() {
  return (
    <svg className="mark" viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="8" fill="#1a3e66" />
      <path d="M8.5 23.2 16 8.2l7.5 15" fill="none" stroke="#f4efe4" strokeWidth="2.3" strokeLinejoin="round" />
      <path d="M11.4 17.8h9.2" stroke="#e2b56a" strokeWidth="2.3" strokeLinecap="round" />
    </svg>
  );
}

export function Shell() {
  const { data } = useProgress();
  const location = useLocation();
  const [menuOpen, setMenuOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const theme = document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
  const [mode, setMode] = useState<"light" | "dark">(theme);

  useEffect(() => {
    setMenuOpen(false);
    window.scrollTo(0, 0);
  }, [location.pathname]);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      const typing =
        target instanceof HTMLElement &&
        (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable);
      if (event.key === "/" && !typing) {
        event.preventDefault();
        setSearchOpen(true);
      }
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setSearchOpen(true);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  function toggleTheme() {
    const next = mode === "dark" ? "light" : "dark";
    setMode(next);
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("aim-theme", next);
  }

  const overall = lessons.length
    ? lessons.reduce((sum, lesson) => sum + lessonFraction(lesson, data), 0) / lessons.length
    : 0;

  return (
    <div className="app">
      <a className="skip" href="#main">
        Skip to content
      </a>
      <header className="topbar">
        <button className="icon-btn menu-btn" type="button" aria-label="Open menu" onClick={() => setMenuOpen(true)}>
          <span />
          <span />
          <span />
        </button>
        <Link to="/" className="brand">
          <Mark />
          <span>AI Mastery</span>
        </Link>
        <button className="search-btn" type="button" onClick={() => setSearchOpen(true)}>
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <circle cx="11" cy="11" r="6.5" fill="none" stroke="currentColor" strokeWidth="1.8" />
            <path d="M16 16.5 20 20.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          </svg>
          <span className="search-label">Search lessons</span>
          <kbd>/</kbd>
        </button>
        <Link to="/dashboard" className="nav-link" aria-label="Progress">
          <span className="nav-word">Progress</span>
          <span className="nav-pct">{Math.round(overall * 100)}%</span>
        </Link>
        <button className="icon-btn" type="button" onClick={toggleTheme} aria-label="Toggle color theme">
          {mode === "dark" ? "Light" : "Dark"}
        </button>
      </header>
      <div className="frame">
        {menuOpen ? (
          <button className="scrim" type="button" aria-label="Close menu" onClick={() => setMenuOpen(false)} />
        ) : null}
        <aside className={menuOpen ? "sidebar open" : "sidebar"}>
          <p className="side-kicker">Curriculum</p>
          {modules.map((module) => (
            <section key={module.id} className="side-module">
              <Link to={`/module/${module.id}`} className="side-module-title">
                <span>0{module.number}</span>
                {module.title}
              </Link>
              <ul>
                {module.lessons.map((lesson) => {
                  const fraction = lessonFraction(lesson, data);
                  const done = Boolean(data.completed[lesson.id]);
                  const active = location.pathname === `/lesson/${lesson.id}`;
                  const state = done ? "done" : fraction > 0 ? "partial" : "empty";
                  return (
                    <li key={lesson.id}>
                      <Link to={`/lesson/${lesson.id}`} className={active ? "lesson-link active" : "lesson-link"}>
                        <span className={`dot ${state}`} aria-hidden="true" />
                        <span>{lesson.title}</span>
                      </Link>
                    </li>
                  );
                })}
              </ul>
            </section>
          ))}
        </aside>
        <main id="main">
          <Outlet />
        </main>
      </div>
      <SearchDialog open={searchOpen} onClose={() => setSearchOpen(false)} />
    </div>
  );
}

import { useEffect } from "react";

export function useTitle(title: string): void {
  useEffect(() => {
    document.title = title === "AI Mastery" ? title : `${title} · AI Mastery`;
  }, [title]);
}

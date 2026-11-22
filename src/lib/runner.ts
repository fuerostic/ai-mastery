export interface RunResult {
  ok: boolean;
  stdout: string;
  error: string;
  trace: string;
}

let worker: Worker | null = null;
let nextId = 1;
let runtimeReady = false;

interface Pending {
  resolve: (result: RunResult) => void;
  timer: number;
}

const pending = new Map<number, Pending>();

export function pythonRuntimeReady(): boolean {
  return runtimeReady;
}

function settleAll(result: RunResult): void {
  for (const [id, item] of pending) {
    window.clearTimeout(item.timer);
    pending.delete(id);
    item.resolve(result);
  }
}

function ensureWorker(): Worker {
  if (worker) return worker;
  const url = `${import.meta.env.BASE_URL}py-worker.js`;
  worker = new Worker(url);
  worker.onmessage = (event: MessageEvent) => {
    const data = (event.data ?? {}) as {
      id?: number;
      runtimeReady?: boolean;
      ok?: boolean;
      stdout?: string;
      error?: string;
      trace?: string;
    };
    if (data.runtimeReady) runtimeReady = true;
    if (typeof data.id !== "number") return;
    const item = pending.get(data.id);
    if (!item) return;
    window.clearTimeout(item.timer);
    pending.delete(data.id);
    item.resolve({
      ok: Boolean(data.ok),
      stdout: String(data.stdout ?? ""),
      error: String(data.error ?? ""),
      trace: String(data.trace ?? ""),
    });
  };
  worker.onerror = (event) => {
    runtimeReady = false;
    worker?.terminate();
    worker = null;
    settleAll({
      ok: false,
      stdout: "",
      error: event.message || "The Python worker failed to start. Reload the page and try again.",
      trace: "",
    });
  };
  return worker;
}

export function runExercise(code: string, tests: string): Promise<RunResult> {
  const firstLoad = !runtimeReady;
  const timeoutMs = firstLoad ? 120_000 : 8_000;
  const id = nextId++;
  const current = ensureWorker();
  return new Promise((resolve) => {
    const timer = window.setTimeout(() => {
      pending.delete(id);
      current.terminate();
      if (worker === current) worker = null;
      runtimeReady = false;
      resolve({
        ok: false,
        stdout: "",
        error: firstLoad
          ? "Python did not finish loading within two minutes. Check your connection and run the tests again."
          : "Stopped the run after 8 seconds. Look for an infinite loop, then run the tests again.",
        trace: "",
      });
    }, timeoutMs);
    pending.set(id, { resolve, timer });
    current.postMessage({ id, code, tests });
  });
}

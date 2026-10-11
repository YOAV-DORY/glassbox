import { useEffect, useState } from "react";

export interface LiveState<T> {
  result: T | null; // last good result, kept on screen while a new one loads
  loading: boolean;
  error: string | null;
}

const DEBOUNCE_MS = 250;

/**
 * Re-runs an API request whenever its inputs change, live.
 * - Debounce: wait until the user pauses for 250 ms instead of calling the API on every
 *   keystroke or slider tick.
 * - AbortController: if the inputs change again, cancel the older request so a slow,
 *   outdated answer can never overwrite a newer one.
 *
 * `request` returns the API call, or a string explaining why it can't run yet
 * (e.g. "Enter a loan amount").
 */
export function useLiveRequest<T>(
  request: (signal: AbortSignal) => Promise<T> | string,
  inputs: unknown[],
): LiveState<T> {
  const [state, setState] = useState<LiveState<T>>({ result: null, loading: true, error: null });

  useEffect(() => {
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      const pending = request(controller.signal);
      if (typeof pending === "string") {
        setState((s) => ({ ...s, loading: false, error: pending }));
        return;
      }
      setState((s) => ({ ...s, loading: true }));
      try {
        const result = await pending;
        setState({ result, loading: false, error: null });
      } catch (error) {
        if (!controller.signal.aborted) {
          setState((s) => ({ ...s, loading: false, error: (error as Error).message }));
        }
      }
    }, DEBOUNCE_MS);

    // Runs before the next effect (inputs changed) or when the component unmounts.
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
    // `request` is a new function on every render, so it is not a dependency;
    // `inputs` lists what it reads, which is what decides when to re-run.
  }, inputs);

  return state;
}

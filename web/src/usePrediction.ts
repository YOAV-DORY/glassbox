import { useEffect, useState } from "react";

import { predict, validateDraft } from "./api";
import type { ApplicantDraft, PredictResponse } from "./types";

export interface PredictionState {
  result: PredictResponse | null; // last good result, kept while a new one loads
  loading: boolean;
  error: string | null;
}

const DEBOUNCE_MS = 250;

/**
 * Re-scores the applicant whenever the form changes, live.
 * - Debounce: wait until the user pauses for 250 ms instead of calling the API per keystroke.
 * - AbortController: if the form changes again, cancel the older request so a slow,
 *   outdated answer can never overwrite a newer one.
 */
export function usePrediction(draft: ApplicantDraft): PredictionState {
  const [state, setState] = useState<PredictionState>({ result: null, loading: true, error: null });

  useEffect(() => {
    const applicant = validateDraft(draft);
    if (typeof applicant === "string") {
      setState((s) => ({ ...s, loading: false, error: applicant }));
      return;
    }

    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      setState((s) => ({ ...s, loading: true }));
      try {
        const result = await predict(applicant, controller.signal);
        setState({ result, loading: false, error: null });
      } catch (error) {
        if (!controller.signal.aborted) {
          setState((s) => ({ ...s, loading: false, error: (error as Error).message }));
        }
      }
    }, DEBOUNCE_MS);

    // Runs before the next effect (form changed) or when the component unmounts.
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [draft]);

  return state;
}

import type { Applicant, ApplicantDraft, ModelInfo, PredictResponse } from "./types";

// Vite forwards /api/* to FastAPI (see vite.config.ts).
const BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(BASE + path, init);
  } catch (error) {
    if (init?.signal?.aborted) throw error; // cancelled on purpose, not a failure
    throw new Error("Can't reach the model API. Is uvicorn running on port 8000?");
  }
  if (!response.ok) {
    // FastAPI validation errors (422) carry a list of {loc, msg}; show the first one.
    const body = await response.json().catch(() => null);
    const first = body?.detail?.[0];
    const message = first ? `${first.loc.at(-1)}: ${first.msg}` : `Request failed (${response.status})`;
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export function fetchModelInfo(): Promise<ModelInfo> {
  return request<ModelInfo>("/model/info");
}

export function predict(applicant: Applicant, signal: AbortSignal): Promise<PredictResponse> {
  return request<PredictResponse>("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ applicant }),
    signal,
  });
}

/** Returns a complete Applicant, or a message saying what is still missing. */
export function validateDraft(draft: ApplicantDraft): Applicant | string {
  if (!draft.amt_credit) return "Enter a loan amount";
  if (!draft.amt_annuity) return "Enter the annuity (regular payment)";
  if (!draft.amt_income_total) return "Enter the applicant's income";
  return { ...draft, amt_credit: draft.amt_credit, amt_annuity: draft.amt_annuity, amt_income_total: draft.amt_income_total };
}

import type {
  Applicant,
  ApplicantDraft,
  ModelInfo,
  Policy,
  PortfolioResponse,
  PredictResponse,
  RiskSurface,
} from "./types";

// Vite forwards /api/* to FastAPI (see vite.config.ts).
const BASE = "/api";

// A free hosting plan puts the server to sleep when nobody uses it; the first request then
// waits while it wakes up (often 30-60 s). Connection failures and "gateway" errors are
// therefore retried for a while before we show an error.
const RETRY_STATUSES = new Set([502, 503, 504]);
const RETRY_FOR_MS = 90_000;
const RETRY_EVERY_MS = 3_000;

const wait = (ms: number, signal?: AbortSignal | null) =>
  new Promise<void>((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(timer);
      reject(signal.reason);
    });
  });

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const giveUpAt = Date.now() + RETRY_FOR_MS;
  for (;;) {
    let response: Response | null = null;
    try {
      response = await fetch(BASE + path, init);
    } catch (error) {
      if (init?.signal?.aborted) throw error; // cancelled on purpose, not a failure
    }
    if (response && !RETRY_STATUSES.has(response.status)) {
      if (!response.ok) {
        // FastAPI validation errors (422) carry a list of {loc, msg}; show the first one.
        const body = await response.json().catch(() => null);
        const first = body?.detail?.[0];
        throw new Error(first ? `${first.loc.at(-1)}: ${first.msg}` : `Request failed (${response.status})`);
      }
      return response.json() as Promise<T>;
    }
    if (Date.now() > giveUpAt) {
      throw new Error("The model server isn't responding. Please try again in a minute.");
    }
    await wait(RETRY_EVERY_MS, init?.signal);
  }
}

export function fetchModelInfo(): Promise<ModelInfo> {
  return request<ModelInfo>("/model/info");
}

export function fetchRiskSurface(): Promise<RiskSurface> {
  return request<RiskSurface>("/model/surface");
}

function post<T>(path: string, body: unknown, signal: AbortSignal): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
}

export function predict(applicant: Applicant, policy: Policy, signal: AbortSignal): Promise<PredictResponse> {
  return post<PredictResponse>("/predict", { applicant, policy }, signal);
}

export function simulatePortfolio(policy: Policy, signal: AbortSignal): Promise<PortfolioResponse> {
  return post<PortfolioResponse>("/portfolio/simulate", policy, signal);
}

/** Returns a complete Applicant, or a message saying what is still missing. */
export function validateDraft(draft: ApplicantDraft): Applicant | string {
  if (!draft.amt_credit) return "Enter a loan amount";
  if (!draft.amt_annuity) return "Enter the annuity (regular payment)";
  if (!draft.amt_income_total) return "Enter the applicant's income";
  return { ...draft, amt_credit: draft.amt_credit, amt_annuity: draft.amt_annuity, amt_income_total: draft.amt_income_total };
}

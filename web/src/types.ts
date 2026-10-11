// Mirrors api/schemas.py. If the API changes, update this file to match.

export const CONTRACT_TYPES = ["Cash loans", "Revolving loans"] as const;
export const INCOME_TYPES = [
  "Working",
  "Commercial associate",
  "State servant",
  "Pensioner",
  "Unemployed",
  "Student",
  "Businessman",
  "Maternity leave",
] as const;
export const EDUCATION_TYPES = [
  "Lower secondary",
  "Secondary / secondary special",
  "Incomplete higher",
  "Higher education",
  "Academic degree",
] as const;

// `typeof LIST[number]` turns each list above into a union type, e.g.
// ContractType = "Cash loans" | "Revolving loans". One source for options and types.
export type ContractType = (typeof CONTRACT_TYPES)[number];
export type IncomeType = (typeof INCOME_TYPES)[number];
export type EducationType = (typeof EDUCATION_TYPES)[number];

export interface Applicant {
  amt_credit: number;
  amt_annuity: number;
  amt_goods_price: number | null;
  contract_type: ContractType;
  amt_income_total: number;
  income_type: IncomeType;
  education_type: EducationType;
  years_employed: number | null; // null = not employed
  ext_source_1: number | null; // null = no score available
  ext_source_2: number | null;
  ext_source_3: number | null;
  own_car: boolean;
  own_realty: boolean;
  cnt_children: number;
  region_rating: 1 | 2 | 3;
}

// While the user is typing, a required amount can be temporarily empty.
// The form edits a draft; only a validated draft becomes an Applicant.
export type ApplicantDraft = Omit<Applicant, "amt_credit" | "amt_annuity" | "amt_income_total"> & {
  amt_credit: number | null;
  amt_annuity: number | null;
  amt_income_total: number | null;
};

/** One factor behind a prediction: its SHAP contribution, in log-odds. */
export interface Reason {
  feature: string;
  label: string; // e.g. "Credit bureau score 2"
  value: string; // e.g. "0.48" or "no score on file"
  impact: number; // > 0 raises risk, < 0 lowers it
}

/** Lending economics, as shares of the loan amount. */
export interface Policy {
  lgd: number; // loss given default: share of the loan lost when a borrower defaults
  margin: number; // profit on a repaid loan
  threshold_override: number | null; // a hand-set cut-off instead of the formula
}

export interface PredictResponse {
  probability_of_default: number;
  decision: "approve" | "decline";
  threshold: number;
  model_version: string;
  expected_profit: number; // of approving this loan, under the policy
  reasons_up: Reason[]; // strongest first; reason-code style, not an adverse-action notice
  reasons_down: Reason[];
}

export interface ModelInfo {
  model_version: string;
  model_type: string;
  dataset: string; // "home-credit", or "synthetic-demo" for the demo model
  trained_at: string;
  train_rows: number;
  validation_roc_auc: number;
  validation_pr_auc: number;
  test_roc_auc: number | null; // held-out test set, scored once after model selection
  test_pr_auc: number | null;
  average_default_rate: number;
  policy: { lgd: number; margin: number; threshold: number };
  global_importance: { feature: string; label: string; importance: number }[];
}

export interface RiskSurface {
  x_feature: string;
  y_feature: string;
  x: number[];
  y: number[];
  z: number[][]; // z[i][j] = probability of default at y[i], x[j]
  threshold: number;
}

/** What a cut-off would have done to the validation applicants, using their real outcomes. */
export interface PortfolioOutcome {
  threshold: number;
  approval_rate: number;
  bad_rate: number; // share of approved loans that defaulted
  approved: number;
  expected_loss: number; // from the model's probabilities alone
  realized_loss: number; // from what actually happened
  expected_profit: number;
  realized_profit: number;
}

export interface CurvePoint {
  threshold: number;
  approval_rate: number;
  bad_rate: number;
  expected_profit: number;
  realized_profit: number;
}

export interface CalibrationBin {
  decile: number; // 1 = lowest predicted risk
  n: number;
  low: number;
  high: number;
  mean_predicted: number;
  actual_rate: number;
  actual_se: number; // standard error of actual_rate
}

export interface FairnessGroup {
  group: string;
  n: number;
  default_rate: number;
  default_rate_se: number;
  mean_predicted: number;
  approval_rate: number;
  default_rate_approved: number | null;
  repayers_declined: number;
  approval_ratio: number; // < 0.8: below the four-fifths rule of thumb
}

export interface PortfolioResponse {
  cohort: "validation";
  applicants: number;
  formula_threshold: number;
  best_threshold: number; // most profitable cut-off on this cohort
  policy: PortfolioOutcome;
  approve_all: PortfolioOutcome;
  cutoff_50: PortfolioOutcome;
  curve: CurvePoint[];
  calibration: CalibrationBin[];
  fairness: { gender: FairnessGroup[]; age_band: FairnessGroup[]; proxy: Record<string, number> };
}

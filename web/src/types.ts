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

export interface PredictResponse {
  probability_of_default: number;
  decision: "approve" | "decline";
  threshold: number;
  model_version: string;
  reasons_up: Reason[]; // strongest first; on a decline these are the principal reasons
  reasons_down: Reason[];
}

export interface ModelInfo {
  model_version: string;
  model_type: string;
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

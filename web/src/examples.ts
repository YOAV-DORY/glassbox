import type { ApplicantDraft } from "./types";

export interface Example {
  id: string;
  label: string;
  description: string;
  applicant: ApplicantDraft;
}

// Made-up applicants that land on different sides of the default policy cut-off.
export const EXAMPLES: Example[] = [
  {
    id: "low",
    label: "Low risk",
    description: "Long tenure, strong scores",
    applicant: {
      amt_credit: 450_000,
      amt_annuity: 21_600,
      amt_goods_price: 450_000,
      contract_type: "Cash loans",
      amt_income_total: 270_000,
      income_type: "State servant",
      education_type: "Higher education",
      years_employed: 12,
      ext_source_1: 0.72,
      ext_source_2: 0.74,
      ext_source_3: 0.7,
      own_car: true,
      own_realty: true,
      cnt_children: 1,
      region_rating: 1,
    },
  },
  {
    id: "borderline",
    label: "Borderline",
    description: "Average scores, thin history",
    applicant: {
      amt_credit: 675_000,
      amt_annuity: 32_800,
      amt_goods_price: 607_500,
      contract_type: "Cash loans",
      amt_income_total: 157_500,
      income_type: "Working",
      education_type: "Secondary / secondary special",
      years_employed: 3.5,
      ext_source_1: null,
      ext_source_2: 0.48,
      ext_source_3: 0.45,
      own_car: false,
      own_realty: true,
      cnt_children: 1,
      region_rating: 2,
    },
  },
  {
    id: "high",
    label: "High risk",
    description: "Weak scores, stretched budget",
    applicant: {
      amt_credit: 1_035_000,
      amt_annuity: 46_300,
      amt_goods_price: 900_000,
      contract_type: "Cash loans",
      amt_income_total: 112_500,
      income_type: "Working",
      education_type: "Lower secondary",
      years_employed: 0.8,
      ext_source_1: null,
      ext_source_2: 0.3,
      ext_source_3: 0.3,
      own_car: false,
      own_realty: false,
      cnt_children: 2,
      region_rating: 3,
    },
  },
];

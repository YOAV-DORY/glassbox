import { useEffect, useState } from "react";

import type { Policy } from "./types";

// Same placeholder economics as the API's defaults (credit_risk/config.py).
export const DEFAULT_POLICY: Policy = { lgd: 0.6, margin: 0.1, threshold_override: null };

/** Same formula as credit_risk/decision.py, so the page can show it instantly while the API works. */
export function formulaThreshold(policy: Policy): number {
  return policy.margin / (policy.margin + policy.lgd);
}

export function activeThreshold(policy: Policy): number {
  return policy.threshold_override ?? formulaThreshold(policy);
}

const STORAGE_KEY = "glassbox.policy";

/**
 * The lending policy, shared by the desk and the simulator. Remembered in this browser
 * (localStorage) so a reload keeps your sliders; falls back to the defaults if storage
 * is unavailable, e.g. in a private window.
 */
export function usePolicy(): [Policy, (policy: Policy) => void] {
  const [policy, setPolicy] = useState<Policy>(() => {
    try {
      const saved = window.localStorage.getItem(STORAGE_KEY);
      return saved ? { ...DEFAULT_POLICY, ...JSON.parse(saved) } : DEFAULT_POLICY;
    } catch {
      return DEFAULT_POLICY;
    }
  });

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(policy));
    } catch {
      // Storage blocked: the policy still works, it just won't survive a reload.
    }
  }, [policy]);

  return [policy, setPolicy];
}

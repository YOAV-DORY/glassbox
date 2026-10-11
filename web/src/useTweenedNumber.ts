import { useEffect, useRef, useState } from "react";

const prefersReducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Animates a displayed number from its current value to `target` (ease-out). */
export function useTweenedNumber(target: number, durationMs = 550): number {
  const [value, setValue] = useState(target);
  const currentRef = useRef(target); // where the animation is right now

  useEffect(() => {
    if (prefersReducedMotion()) {
      currentRef.current = target;
      setValue(target);
      return;
    }
    const from = currentRef.current;
    const start = performance.now();
    let frame = 0;

    const step = (now: number) => {
      const t = Math.min((now - start) / durationMs, 1);
      const eased = 1 - (1 - t) ** 3; // fast start, gentle landing
      currentRef.current = from + (target - from) * eased;
      setValue(currentRef.current);
      if (t < 1) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [target, durationMs]);

  return value;
}

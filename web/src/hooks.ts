import { useEffect, useRef, useState, type RefObject } from "react";

export const prefersReducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** True once the element has scrolled into view (and stays true: reveal animations play once). */
export function useInView<T extends Element>(threshold = 0.2): [RefObject<T | null>, boolean] {
  const ref = useRef<T>(null);
  const [inView, setInView] = useState(false);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setInView(true);
          observer.disconnect();
        }
      },
      { threshold },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, [threshold]);
  return [ref, inView];
}

/** True while the element is on screen. Used to pause canvas animations that nobody can see. */
export function useOnScreen<T extends Element>(ref: RefObject<T | null>): boolean {
  const [onScreen, setOnScreen] = useState(false);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new IntersectionObserver(([entry]) => setOnScreen(entry.isIntersecting));
    observer.observe(element);
    return () => observer.disconnect();
  }, [ref]);
  return onScreen;
}

/**
 * Calls `apply(progress)` as the element travels through the viewport:
 * 0 when its top enters at the bottom of the screen, 1 when its bottom leaves at the top.
 * It writes styles directly instead of setting React state, so scrolling doesn't
 * re-render components 60 times a second.
 */
export function useScrollProgress<T extends HTMLElement>(ref: RefObject<T | null>, apply: (progress: number) => void) {
  const applyRef = useRef(apply);
  applyRef.current = apply;

  useEffect(() => {
    let frame = 0;
    const update = () => {
      frame = 0;
      const element = ref.current;
      if (!element) return;
      const rect = element.getBoundingClientRect();
      const total = window.innerHeight + rect.height;
      const progress = Math.min(Math.max((window.innerHeight - rect.top) / total, 0), 1);
      applyRef.current(progress);
    };
    // At most one update per animation frame, however many scroll events fire.
    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(update);
    };
    update();
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
    };
  }, [ref]);
}

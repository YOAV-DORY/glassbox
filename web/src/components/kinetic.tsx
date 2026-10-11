import { Fragment, useEffect, useRef, useState, type CSSProperties, type ElementType, type ReactNode } from "react";

import { prefersReducedMotion, useInView, useScrollProgress } from "../hooks";
import { useTweenedNumber } from "../useTweenedNumber";

/**
 * Text that builds itself letter by letter: each character flips up into place, slightly
 * after the one before it. Screen readers get the plain text via aria-label.
 */
export function KineticText(props: { text: string; delayMs?: number; stepMs?: number; className?: string }) {
  const { text, delayMs = 0, stepMs = 32 } = props;
  return (
    <span className={`kinetic ${props.className ?? ""}`} aria-label={text}>
      {text.split(" ").map((word, w, words) => {
        const before = words.slice(0, w).join(" ").length + (w ? 1 : 0); // letters before this word
        return (
          <Fragment key={w}>
            {/* Words never break in the middle: each one is an unbreakable inline block. */}
            <span className="kinetic-word" aria-hidden="true">
              {Array.from(word).map((char, c) => (
                <span key={c} className="kinetic-char" style={{ "--d": `${delayMs + (before + c) * stepMs}ms` } as CSSProperties}>
                  {char}
                </span>
              ))}
            </span>
            {w < words.length - 1 && " "}
          </Fragment>
        );
      })}
    </span>
  );
}

/** Like KineticText, but the letters only start flipping in once the text scrolls into view. */
export function KineticOnView(props: { text: string; className?: string; stepMs?: number }) {
  const [ref, inView] = useInView<HTMLSpanElement>(0.4);
  return (
    <span ref={ref}>
      {inView ? (
        <KineticText text={props.text} className={props.className} stepMs={props.stepMs} />
      ) : (
        // Invisible stand-in with the same text, so the layout doesn't jump when it animates in.
        <span className={`kinetic-placeholder ${props.className ?? ""}`}>{props.text}</span>
      )}
    </span>
  );
}

/** Cycles through words, each one flipping in letter by letter. */
export function WordCycler(props: { words: { text: string; className: string }[]; intervalMs?: number; delayMs?: number }) {
  const { words, intervalMs = 2600, delayMs = 0 } = props;
  const [index, setIndex] = useState(0);
  const [first, setFirst] = useState(true);

  useEffect(() => {
    if (prefersReducedMotion()) return;
    const timer = window.setInterval(() => {
      setFirst(false);
      setIndex((i) => (i + 1) % words.length);
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [words.length, intervalMs]);

  const word = words[index];
  return (
    <span className="cycler" aria-live="off">
      {/* key: a new word remounts, which replays the letter animation */}
      <KineticText key={word.text} text={word.text} className={word.className} delayMs={first ? delayMs : 0} stepMs={45} />
    </span>
  );
}

/** Giant words that slide sideways as the page scrolls (scroll-linked, not time-based). */
export function Marquee(props: { items: string[]; direction?: 1 | -1 }) {
  const ref = useRef<HTMLDivElement>(null);
  const trackRef = useRef<HTMLDivElement>(null);
  const direction = props.direction ?? 1;
  useScrollProgress(ref, (progress) => {
    if (trackRef.current) trackRef.current.style.transform = `translate3d(${(progress - 0.5) * 30 * direction}%, 0, 0)`;
  });
  const items = [...props.items, ...props.items, ...props.items]; // wide enough to never show an edge
  return (
    <div ref={ref} className="marquee" aria-hidden="true">
      <div ref={trackRef} className="marquee-track">
        {items.map((item, i) => (
          <span key={i} className={i % 2 ? "marquee-item is-filled" : "marquee-item"}>
            {item}
            <i className="marquee-dot" />
          </span>
        ))}
      </div>
    </div>
  );
}

/** Fades and lifts its children into place the first time they scroll into view. */
export function Reveal(props: { as?: ElementType; delayMs?: number; className?: string; children: ReactNode }) {
  const [ref, inView] = useInView<HTMLElement>(0.15);
  const Tag = props.as ?? "div";
  return (
    <Tag
      ref={ref}
      className={`scroll-reveal ${inView ? "is-in" : ""} ${props.className ?? ""}`}
      style={{ "--delay": `${props.delayMs ?? 0}ms` } as CSSProperties}
    >
      {props.children}
    </Tag>
  );
}

/** A number that counts up from zero when it scrolls into view. */
export function CountUp(props: { value: number; format: (value: number) => string }) {
  const [ref, inView] = useInView<HTMLSpanElement>(0.4);
  const shown = useTweenedNumber(inView ? props.value : 0, 1600);
  return (
    <span ref={ref} className="count-up">
      {props.format(shown)}
    </span>
  );
}

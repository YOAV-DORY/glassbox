import { useEffect, useRef } from "react";

import { prefersReducedMotion, useOnScreen, useScrollProgress } from "../hooks";
import { COLORS, cubeEdges, depthFade, fitCanvas, line, project, seededRandom, type Vec3, type View } from "../wire3d";

// The logo in 3D: an ink box and an offset cobalt "glass" box, with applicants floating
// inside. A scoring plane sweeps through and lights up the applicants it passes.
const OUTER = cubeEdges(0.9);
const INNER = cubeEdges(0.55, [0.38, -0.32, 0.38]);
const random = seededRandom(7);
const APPLICANTS = Array.from({ length: 90 }, (_, i) => ({
  position: [random() * 1.6 - 0.8, random() * 1.6 - 0.8, random() * 1.6 - 0.8] as Vec3,
  defaulted: i % 12 === 0, // ~8%, like the real data
  phase: random() * Math.PI * 2,
}));
const FLOOR_LINES: [Vec3, Vec3][] = [];
for (let v = -1.6; v <= 1.601; v += 0.4) {
  FLOOR_LINES.push([[v, -1.35, -1.6], [v, -1.35, 1.6]], [[-1.6, -1.35, v], [1.6, -1.35, v]]);
}

export function WireCube() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const onScreen = useOnScreen(canvasRef);
  const pointer = useRef({ x: 0, y: 0 }); // -1..1 across the window
  const scroll = useRef(0);
  useScrollProgress(canvasRef, (progress) => (scroll.current = progress));

  useEffect(() => {
    const move = (event: PointerEvent) => {
      pointer.current = { x: (event.clientX / window.innerWidth) * 2 - 1, y: (event.clientY / window.innerHeight) * 2 - 1 };
    };
    window.addEventListener("pointermove", move);
    return () => window.removeEventListener("pointermove", move);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !onScreen) return; // nothing to animate while off screen
    const still = prefersReducedMotion();
    const start = performance.now();
    const eased = { x: 0, y: 0 }; // pointer position, smoothed so the box drifts instead of jumping
    let frame = 0;

    const draw = (now: number) => {
      const t = still ? 0 : (now - start) / 1000;
      eased.x += (pointer.current.x - eased.x) * 0.04;
      eased.y += (pointer.current.y - eased.y) * 0.04;
      const { ctx, width, height } = fitCanvas(canvas);
      const view: View = {
        yaw: 0.65 + t * 0.16 + scroll.current * 2.4 + eased.x * 0.4,
        pitch: 0.42 + eased.y * 0.15,
        scale: Math.min(width, height) * 0.25, // leaves room for perspective: near corners draw larger
        cx: width / 2,
        cy: height / 2,
      };
      drawScene(ctx, view, t);
      if (!still) frame = requestAnimationFrame(draw);
    };
    frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, [onScreen]);

  return <canvas ref={canvasRef} className="wire-cube" aria-hidden="true" />;
}

function drawScene(ctx: CanvasRenderingContext2D, view: View, t: number) {
  const drawEdges = (edges: [Vec3, Vec3][], color: string, width: number, strength = 1) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    for (const [a, b] of edges) {
      const pa = project(a, view);
      const pb = project(b, view);
      ctx.globalAlpha = depthFade((pa.depth + pb.depth) / 2) * strength;
      line(ctx, pa, pb);
    }
  };

  drawEdges(FLOOR_LINES, COLORS.ink, 1, 0.16);

  // The scoring plane: a square that rises and falls through the box.
  const planeY = Math.sin(t * 0.7) * 0.75;
  const s = 0.9;
  const plane: Vec3[] = [[-s, planeY, -s], [s, planeY, -s], [s, planeY, s], [-s, planeY, s]];
  const corners = plane.map((p) => project(p, view));
  ctx.globalAlpha = 0.08;
  ctx.fillStyle = COLORS.cobalt;
  ctx.beginPath();
  corners.forEach((c, i) => (i ? ctx.lineTo(c.x, c.y) : ctx.moveTo(c.x, c.y)));
  ctx.closePath();
  ctx.fill();
  ctx.setLineDash([4, 5]);
  drawEdges(plane.map((p, i) => [p, plane[(i + 1) % 4]] as [Vec3, Vec3]), COLORS.cobalt, 1.2, 0.8);
  ctx.setLineDash([]);

  // Applicants: dots that bob gently. Dots near the plane are "being scored" and light up.
  for (const applicant of APPLICANTS) {
    const [x, y, z] = applicant.position;
    const bobbed: Vec3 = [x, y + Math.sin(t * 0.8 + applicant.phase) * 0.04, z];
    const p = project(bobbed, view);
    const scored = Math.abs(bobbed[1] - planeY) < 0.07;
    ctx.globalAlpha = depthFade(p.depth, 0.9, 0.3);
    ctx.fillStyle = applicant.defaulted ? COLORS.vermilion : scored ? COLORS.cobalt : COLORS.ink;
    ctx.beginPath();
    ctx.arc(p.x, p.y, scored ? 3.4 : applicant.defaulted ? 2.6 : 1.7, 0, Math.PI * 2);
    ctx.fill();
  }

  drawEdges(OUTER, COLORS.ink, 1.4);
  drawEdges(INNER, COLORS.cobalt, 1.6);
  ctx.globalAlpha = 1;
}

import { useEffect, useRef, type PointerEvent } from "react";

import { formatPercent } from "../format";
import { prefersReducedMotion, useOnScreen } from "../hooks";
import type { RiskSurface } from "../types";
import { COLORS, depthFade, fitCanvas, line, project, type Vec3, type View } from "../wire3d";

const HEIGHT = 1.5; // world units from lowest to highest risk
const SPAN = 2.2; // world units across each score axis
const RISK_CAP = 0.4; // heights stop at 40% default risk

/**
 * The real model's predicted default risk over two credit-bureau scores, drawn as a
 * wireframe landscape. Lines above the policy cut-off are vermilion (decline zone).
 * Drag to turn it; it slowly turns on its own otherwise.
 */
export function RiskLandscape({ surface }: { surface: RiskSurface | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const onScreen = useOnScreen(canvasRef);
  const angle = useRef({ yaw: -0.75, pitch: 0.58 });
  const drag = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !surface || !onScreen) return;
    const still = prefersReducedMotion();
    let frame = 0;
    let last = performance.now();

    const draw = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;
      if (!drag.current && !still) angle.current.yaw += dt * 0.12; // idle rotation
      const { ctx, width, height } = fitCanvas(canvas);
      const view: View = {
        ...angle.current,
        scale: Math.min(width, height) * 0.27, // box diagonal stays inside the frame at any angle
        cx: width / 2,
        cy: height * 0.52,
        distance: 8, // gentle perspective: this is a chart, it should stay readable
      };
      drawLandscape(ctx, surface, view);
      frame = requestAnimationFrame(draw); // keep drawing so dragging works even without idle motion
    };
    frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, [surface, onScreen]);

  const onPointerDown = (event: PointerEvent<HTMLCanvasElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId); // keep receiving moves outside the canvas
    drag.current = { x: event.clientX, y: event.clientY };
  };
  const onPointerMove = (event: PointerEvent<HTMLCanvasElement>) => {
    if (!drag.current) return;
    angle.current.yaw += (event.clientX - drag.current.x) * 0.008;
    angle.current.pitch = Math.min(Math.max(angle.current.pitch + (event.clientY - drag.current.y) * 0.006, 0.1), 1.2);
    drag.current = { x: event.clientX, y: event.clientY };
  };
  const onPointerUp = () => (drag.current = null);

  return (
    <canvas
      ref={canvasRef}
      className="landscape-canvas"
      role="img"
      aria-label="3D surface of predicted default risk across two credit bureau scores"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
    />
  );
}

function drawLandscape(ctx: CanvasRenderingContext2D, surface: RiskSurface, view: View) {
  const { x, y, z, threshold } = surface;
  // Same 0-40% range as the decision panel's ruler. A few extreme corners (very low scores)
  // go higher; capping them keeps the rest of the landscape from looking flat.
  const height = (p: number) => (Math.min(p, RISK_CAP) / RISK_CAP) * HEIGHT - HEIGHT / 2;
  const point = (i: number, j: number): Vec3 => [(x[j] - 0.5) * SPAN, height(z[i][j]), (y[i] - 0.5) * SPAN];

  const stroke = (a: Vec3, b: Vec3, color: string, width: number, strength = 1) => {
    const pa = project(a, view);
    const pb = project(b, view);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.globalAlpha = depthFade((pa.depth + pb.depth) / 2) * strength;
    line(ctx, pa, pb);
  };

  // Bounding box: floor square and four corner posts.
  const h = SPAN / 2;
  const floor = -HEIGHT / 2;
  const corners: Vec3[] = [[-h, floor, -h], [h, floor, -h], [h, floor, h], [-h, floor, h]];
  corners.forEach((c, k) => {
    stroke(c, corners[(k + 1) % 4], COLORS.ink, 1, 0.35);
    stroke(c, [c[0], HEIGHT / 2, c[2]], COLORS.ink, 1, 0.15);
  });

  // Policy cut-off: a dashed plane at the threshold height.
  const cut = height(threshold);
  const plane: Vec3[] = corners.map(([cx, , cz]) => [cx, cut, cz]);
  ctx.setLineDash([5, 6]);
  plane.forEach((c, k) => stroke(c, plane[(k + 1) % 4], COLORS.cobalt, 1.3, 0.9));
  ctx.setLineDash([]);

  // The surface: one polyline per row and per column of the grid.
  const n = x.length;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      for (const [di, dj] of [[0, 1], [1, 0]]) {
        const i2 = i + di;
        const j2 = j + dj;
        if (i2 >= n || j2 >= n) continue;
        const decline = (z[i][j] + z[i2][j2]) / 2 >= threshold;
        stroke(point(i, j), point(i2, j2), decline ? COLORS.vermilion : COLORS.ink, 1.1);
      }
    }
  }

  // Labels
  ctx.globalAlpha = 1;
  ctx.font = '11px "IBM Plex Mono", monospace';
  ctx.fillStyle = COLORS.ink;
  const label = (text: string, at: Vec3, color: string = COLORS.ink) => {
    const p = project(at, view);
    ctx.fillStyle = color;
    ctx.fillText(text, p.x + 6, p.y);
  };
  label("Bureau score 2 →", [h * 0.35, floor, -h * 1.12]);
  label("Bureau score 3 →", [-h * 1.12, floor, h * 0.35]);
  label("Default risk ↑ (to 40%)", [-h, HEIGHT / 2, -h]);
  label(`cut-off ${formatPercent(threshold)}`, [h, cut, -h], COLORS.cobalt);
}

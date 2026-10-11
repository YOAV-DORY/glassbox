// A tiny wireframe 3D renderer for <canvas>. No library: rotate points, apply
// perspective, draw lines. That's all a wireframe needs.

export type Vec3 = readonly [number, number, number];

// Same colours as the CSS tokens in styles/base.css (a canvas can't read CSS variables cheaply).
export const COLORS = { ink: "#15171c", cobalt: "#2443d6", vermilion: "#c0391b", muted: "#8a8d94" };

export interface View {
  yaw: number; // rotation around the vertical axis (turning left/right), in radians
  pitch: number; // tilt towards / away from the viewer, in radians
  scale: number; // pixels per world unit
  cx: number; // screen position of the world origin
  cy: number;
  distance?: number; // camera distance: smaller = stronger perspective (default 4)
}

export interface Projected {
  x: number;
  y: number;
  depth: number; // larger = further away
}

const CAMERA_DISTANCE = 4; // how far the eye sits from the origin, in world units

export function project([x, y, z]: Vec3, view: View): Projected {
  // 1. Turn around the vertical axis.
  const cosYaw = Math.cos(view.yaw);
  const sinYaw = Math.sin(view.yaw);
  const x1 = x * cosYaw - z * sinYaw;
  const z1 = x * sinYaw + z * cosYaw;
  // 2. Tilt around the horizontal axis.
  const cosPitch = Math.cos(view.pitch);
  const sinPitch = Math.sin(view.pitch);
  const y2 = y * cosPitch - z1 * sinPitch;
  const z2 = y * sinPitch + z1 * cosPitch;
  // 3. Perspective: things further away are drawn smaller and closer to the centre.
  const distance = view.distance ?? CAMERA_DISTANCE;
  const perspective = distance / (distance + z2);
  return {
    x: view.cx + x1 * perspective * view.scale,
    y: view.cy - y2 * perspective * view.scale, // screen y grows downwards
    depth: z2,
  };
}

/** Opacity that fades lines towards the back, which makes the 3D shape easier to read. */
export function depthFade(depth: number, near = 0.95, far = 0.25): number {
  const t = Math.min(Math.max((depth + 1.5) / 3, 0), 1);
  return near + (far - near) * t;
}

export function line(ctx: CanvasRenderingContext2D, a: Projected, b: Projected) {
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
}

/** Match the canvas's pixel buffer to its on-screen size, so lines stay sharp on retina screens. */
export function fitCanvas(canvas: HTMLCanvasElement): { ctx: CanvasRenderingContext2D; width: number; height: number } {
  const ctx = canvas.getContext("2d")!;
  const ratio = window.devicePixelRatio || 1;
  const { width, height } = canvas.getBoundingClientRect();
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0); // draw in CSS pixels from here on
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}

// The 12 edges of a cube from -s to +s, as pairs of corner points.
export function cubeEdges(s: number, offset: Vec3 = [0, 0, 0]): [Vec3, Vec3][] {
  const [ox, oy, oz] = offset;
  const corner = (i: number): Vec3 => [
    ox + (i & 1 ? s : -s),
    oy + (i & 2 ? s : -s),
    oz + (i & 4 ? s : -s),
  ];
  const edges: [Vec3, Vec3][] = [];
  for (let i = 0; i < 8; i++) {
    for (const bit of [1, 2, 4]) {
      if (!(i & bit)) edges.push([corner(i), corner(i | bit)]); // corners that differ in one axis
    }
  }
  return edges;
}

/** Deterministic pseudo-random numbers, so the decoration looks the same on every visit. */
export function seededRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return state / 4294967296;
  };
}

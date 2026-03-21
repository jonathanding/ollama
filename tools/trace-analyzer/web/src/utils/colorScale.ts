export const BACKEND_COLORS: Record<string, string> = {
  CPU: '#3b82f6',
  CUDA0: '#22c55e',
  CUDA1: '#16a34a',
  Vulkan: '#f97316',
  Vulkan0: '#f97316',
  Metal: '#a855f7',
};

export function backendColor(backend: string): string {
  return BACKEND_COLORS[backend] ?? '#6b7280';
}

export const OP_COLORS: Record<string, string> = {
  MUL_MAT: '#ef4444',
  FLASH_ATTN_EXT: '#f97316',
  RMS_NORM: '#eab308',
  MUL: '#22c55e',
  ADD: '#14b8a6',
  ROPE: '#06b6d4',
  SET_ROWS: '#3b82f6',
  GLU: '#8b5cf6',
  CPY: '#ec4899',
  DUP: '#ec4899',
  GET_ROWS: '#a855f7',
  RESHAPE: '#6b7280',
  PERMUTE: '#9ca3af',
  VIEW: '#d1d5db',
  SOFT_MAX: '#f59e0b',
  CONT: '#d1d5db',
};

const FALLBACK_OP_COLORS = [
  '#64748b', '#78716c', '#71717a', '#737373', '#525252',
];

export function opColor(op: string): string {
  if (OP_COLORS[op]) return OP_COLORS[op];
  let hash = 0;
  for (let i = 0; i < op.length; i++) hash = (hash * 31 + op.charCodeAt(i)) | 0;
  return FALLBACK_OP_COLORS[Math.abs(hash) % FALLBACK_OP_COLORS.length];
}

/**
 * Heatmap: cool-to-warm with light tones for black text readability.
 * Light blue (cold/fast, 0) → Light purple → Light yellow → Light orange (hot/slow, 1).
 * All stops keep lightness high enough for WCAG AA black text contrast.
 */
export function heatmapColor(ratio: number): string {
  const t = Math.max(0, Math.min(1, ratio));
  // 5-stop: light-blue → light-purple → light-yellow → light-orange → warm-coral
  let r: number, g: number, b: number;
  if (t < 0.25) {
    // #dbeafe (blue-100) → #e9d5ff (purple-200)
    const s = t / 0.25;
    r = Math.round(219 + s * (233 - 219));
    g = Math.round(234 - s * (234 - 213));
    b = Math.round(254 - s * (254 - 255));
  } else if (t < 0.5) {
    // #e9d5ff (purple-200) → #fef9c3 (yellow-100)
    const s = (t - 0.25) / 0.25;
    r = Math.round(233 + s * (254 - 233));
    g = Math.round(213 + s * (249 - 213));
    b = Math.round(255 - s * (255 - 195));
  } else if (t < 0.75) {
    // #fef9c3 (yellow-100) → #fed7aa (orange-200)
    const s = (t - 0.5) / 0.25;
    r = Math.round(254);
    g = Math.round(249 - s * (249 - 215));
    b = Math.round(195 - s * (195 - 170));
  } else {
    // #fed7aa (orange-200) → #fca5a5 (red-300)
    const s = (t - 0.75) / 0.25;
    r = Math.round(254 - s * (254 - 252));
    g = Math.round(215 - s * (215 - 165));
    b = Math.round(170 - s * (170 - 165));
  }
  return `rgb(${r},${g},${b})`;
}

/** Precomputed heatmap stops for legend / histogram display */
export const HEATMAP_STOPS = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0].map(t => ({
  ratio: t,
  color: heatmapColor(t),
}));

export function diffColor(diffPct: number, threshold: number = 10): string {
  if (diffPct > threshold) return '#fee2e2';
  if (diffPct < -threshold) return '#dcfce7';
  return 'transparent';
}

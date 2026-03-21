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

export function heatmapColor(ratio: number): string {
  const r = ratio < 0.5 ? Math.round(ratio * 2 * 255) : 255;
  const g = ratio < 0.5 ? 255 : Math.round((1 - (ratio - 0.5) * 2) * 255);
  const b = ratio < 0.5 ? Math.round((1 - ratio * 2) * 255) : 0;
  return `rgb(${r},${g},${b})`;
}

export function diffColor(diffPct: number, threshold: number = 10): string {
  if (diffPct > threshold) return '#fee2e2';
  if (diffPct < -threshold) return '#dcfce7';
  return 'transparent';
}

export const BACKEND_COLORS: Record<string, string> = {
  CPU: '#3b82f6',
  CUDA0: '#22c55e',
  CUDA1: '#16a34a',
  Vulkan: '#f97316',
  Metal: '#a855f7',
};

export function backendColor(backend: string): string {
  return BACKEND_COLORS[backend] ?? '#6b7280';
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

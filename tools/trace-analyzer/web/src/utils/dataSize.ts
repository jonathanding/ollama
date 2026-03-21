const DTYPE_SIZE: Record<string, number> = {
  f32: 4, f16: 2, bf16: 2,
  q4_0: 0.5, q4_1: 0.5, q5_0: 0.625, q5_1: 0.625,
  q8_0: 1, q8_1: 1, i8: 1, i16: 2, i32: 4,
};

export function estimateBytes(shape: number[], dtype: string): number {
  const size = DTYPE_SIZE[dtype] ?? 2;
  return shape.reduce((a, b) => a * b, 1) * size;
}

export function formatBytes(bytes: number): string {
  if (bytes >= 1_073_741_824) return `${(bytes / 1_073_741_824).toFixed(1)} GB`;
  if (bytes >= 1_048_576) return `${(bytes / 1_048_576).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

export function formatNs(ns: number): string {
  if (ns >= 1_000_000_000) return `${(ns / 1_000_000_000).toFixed(2)}s`;
  if (ns >= 1_000_000) return `${(ns / 1_000_000).toFixed(1)}ms`;
  if (ns >= 1_000) return `${(ns / 1_000).toFixed(0)}us`;
  return `${ns}ns`;
}

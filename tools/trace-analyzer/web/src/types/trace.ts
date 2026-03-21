export interface SummaryData {
  meta: {
    source_file: string;
    model: string | null;
    total_ops: number;
    total_passes: number;
    total_wall_ms: number;
  };
  timing: {
    total_ms: number;
    prefill_ms: number;
    prefill_tokens: number;
    decode_avg_ms: number;
    per_pass: Array<{ pass: number; n_tokens: number; wall_ms: number | null; n_ops: number }>;
  };
  op_stats: Array<{
    op: string; count: number; total_ns: number;
    pct_time: number; avg_ns: number; est_bytes_total?: number;
  }>;
  backend_stats: Array<{
    backend: string; count: number; total_ns: number;
    pct_ops: number; pct_time: number;
  }>;
  copy_stats: {
    count: number; total_ns: number; est_total_bytes: number;
    copies: Array<{
      name: string; op: string; est_bytes: number; ns: number; backend: string;
    }>;
  };
  layer_stats: Array<{
    layer: string; n_ops: number; total_ns: number;
    pct_time: number; top_op: string;
  }>;
  dag: DagData;
  timeline_ops: TimelineOp[];
}

export interface TimelineOp {
  pass: number; seq: number; name: string;
  op: string; backend: string; t_start: number; t_end: number;
}

export interface DagData {
  nodes: DagNode[];
  edges: DagEdge[];
}

export interface DagNode {
  id: string; op: string; backend: string;
  ns: number; shape: number[]; dtype: string;
  layer: string | null; is_copy: boolean;
}

export interface DagEdge {
  from: string; to: string; est_bytes: number;
}

export interface CompareData {
  labels: [string, string];
  meta: Array<{ label: string; source_file: string; model: string | null; total_wall_ms: number }>;
  timing_diff: {
    prefill_ms: [number, number];
    decode_avg_ms: [number, number];
    total_ms: [number, number];
    diff_pct: { prefill: number; decode: number; total: number };
  };
  op_diff: Array<{
    op: string;
    values: Array<{ label: string; total_ns: number; count?: number }>;
    diff_pct: number;
    significant: boolean;
  }>;
  layer_diff: Array<{
    layer: string;
    values: Array<{ label: string; total_ns: number }>;
    diff_pct: number;
    significant: boolean;
  }>;
  copy_diff: Array<{
    name: string;
    values: Array<{ label: string; ns: number; est_bytes: number }>;
    diff_pct: number;
    significant: boolean;
  }>;
}

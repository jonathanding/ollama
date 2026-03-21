import type { DagNode } from '../types/trace';
import { formatNs, formatBytes, estimateBytes } from '../utils/dataSize';

interface Props { node: DagNode; onClose: () => void; }

export function NodeDetail({ node, onClose }: Props) {
  return (
    <>
      <div className="flex items-center justify-between mb-2">
        <h3 className="font-bold text-sm font-mono text-gray-800 truncate">{node.id}</h3>
        <button onClick={onClose} className="text-gray-300 hover:text-gray-500 ml-4 text-lg leading-none transition-colors">&times;</button>
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1.5 text-sm">
        <span className="inline-flex items-center gap-1.5">
          <span className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold">Op</span>
          <span className="font-medium text-gray-700">{node.op}</span>
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold">Backend</span>
          <span className="bg-gray-100 px-1.5 py-0.5 rounded text-xs text-gray-600">{node.backend}</span>
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold">Time</span>
          <span className="font-mono tabular-nums text-gray-700">{formatNs(node.ns)}</span>
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold">Shape</span>
          <span className="font-mono text-gray-600 text-xs">[{node.shape.join(', ')}]</span>
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold">Dtype</span>
          <span className="font-mono text-gray-600 text-xs">{node.dtype}</span>
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold">Size</span>
          <span className="font-mono text-gray-600 text-xs">{formatBytes(estimateBytes(node.shape, node.dtype))}</span>
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold">Layer</span>
          <span className="text-gray-600 text-xs">{node.layer ?? '(top-level)'}</span>
        </span>
        {node.is_copy && <span className="text-red-600 font-medium text-xs bg-red-50 px-2 py-0.5 rounded-md ring-1 ring-red-200">Copy Op</span>}
      </div>
    </>
  );
}

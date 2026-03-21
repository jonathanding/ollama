import type { DagNode } from '../types/trace';
import { formatNs, formatBytes, estimateBytes } from '../utils/dataSize';

interface Props { node: DagNode; onClose: () => void; }

export function NodeDetail({ node, onClose }: Props) {
  return (
    <>
      <div className="flex items-start justify-between mb-2">
        <h3 className="font-bold text-sm font-mono">{node.id}</h3>
        <button onClick={onClose} className="text-gray-400 hover:text-gray-600 ml-4 text-sm">X</button>
      </div>
      <div className="flex flex-wrap gap-x-6 gap-y-1 text-sm">
        <span><span className="text-gray-500">Op:</span> {node.op}</span>
        <span><span className="text-gray-500">Backend:</span> {node.backend}</span>
        <span><span className="text-gray-500">Time:</span> {formatNs(node.ns)}</span>
        <span><span className="text-gray-500">Shape:</span> [{node.shape.join(', ')}]</span>
        <span><span className="text-gray-500">Dtype:</span> {node.dtype}</span>
        <span><span className="text-gray-500">Est. Size:</span> {formatBytes(estimateBytes(node.shape, node.dtype))}</span>
        <span><span className="text-gray-500">Layer:</span> {node.layer ?? '(top-level)'}</span>
        {node.is_copy && <span className="text-red-600 font-medium">Copy Op</span>}
      </div>
    </>
  );
}

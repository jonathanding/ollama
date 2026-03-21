import type { DagNode } from '../types/trace';
import { formatNs, formatBytes, estimateBytes } from '../utils/dataSize';

interface Props { node: DagNode | null; onClose: () => void; }

export function NodeDetail({ node, onClose }: Props) {
  if (!node) return null;
  return (
    <div className="fixed right-72 top-0 h-full w-80 bg-white shadow-lg p-4 overflow-y-auto z-50 border-r">
      <button onClick={onClose} className="float-right text-gray-400 hover:text-gray-600">X</button>
      <h3 className="font-bold text-lg mb-4">{node.id}</h3>
      <table className="w-full text-sm">
        <tbody>
          <tr><td className="text-gray-500 pr-4">Op</td><td>{node.op}</td></tr>
          <tr><td className="text-gray-500 pr-4">Backend</td><td>{node.backend}</td></tr>
          <tr><td className="text-gray-500 pr-4">Time</td><td>{formatNs(node.ns)}</td></tr>
          <tr><td className="text-gray-500 pr-4">Shape</td><td>[{node.shape.join(', ')}]</td></tr>
          <tr><td className="text-gray-500 pr-4">Dtype</td><td>{node.dtype}</td></tr>
          <tr><td className="text-gray-500 pr-4">Est. Size</td><td>{formatBytes(estimateBytes(node.shape, node.dtype))}</td></tr>
          <tr><td className="text-gray-500 pr-4">Layer</td><td>{node.layer ?? '(top-level)'}</td></tr>
          <tr><td className="text-gray-500 pr-4">Copy?</td><td>{node.is_copy ? 'Yes' : 'No'}</td></tr>
        </tbody>
      </table>
    </div>
  );
}

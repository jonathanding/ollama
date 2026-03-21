import { useState } from 'react';
import type { SummaryData, DagNode } from '../types/trace';
import { formatNs } from '../utils/dataSize';

type Tab = 'ops' | 'copies_size' | 'copies_time';

interface Props {
  data: SummaryData;
  selectedId: string | null;
  onSelect: (id: string) => void;
}

export function HotspotPanel({ data, selectedId, onSelect }: Props) {
  const [tab, setTab] = useState<Tab>('ops');
  const [collapsed, setCollapsed] = useState(false);

  if (collapsed) {
    return (
      <button
        className="fixed right-0 top-1/2 bg-gray-800 text-white px-2 py-4 rounded-l z-40"
        onClick={() => setCollapsed(false)}
      >{'<'}</button>
    );
  }

  const opsRanked = [...data.dag.nodes].sort((a, b) => b.ns - a.ns);
  const copies = data.dag.nodes.filter(n => n.is_copy);
  const copiesBySize = [...copies].sort((a, b) => {
    const sa = a.shape.reduce((x, y) => x * y, 1);
    const sb = b.shape.reduce((x, y) => x * y, 1);
    return sb - sa;
  });
  const copiesByTime = [...copies].sort((a, b) => b.ns - a.ns);

  const items: DagNode[] =
    tab === 'ops' ? opsRanked :
    tab === 'copies_size' ? copiesBySize : copiesByTime;

  return (
    <div className="fixed right-0 top-0 h-full w-72 bg-white border-l shadow z-40 flex flex-col">
      <div className="flex items-center justify-between p-2 border-b">
        <span className="font-bold text-sm">Hotspots</span>
        <button onClick={() => setCollapsed(true)} className="text-gray-400">{'>'}</button>
      </div>
      <div className="flex gap-1 p-2 bg-gray-50">
        {(['ops', 'copies_size', 'copies_time'] as Tab[]).map(t => (
          <button
            key={t}
            className={`px-2 py-1 text-xs rounded ${tab === t ? 'bg-gray-800 text-white' : 'bg-gray-200'}`}
            onClick={() => setTab(t)}
          >{t === 'ops' ? 'Top Ops' : t === 'copies_size' ? 'Copies (size)' : 'Copies (time)'}</button>
        ))}
      </div>
      <div className="overflow-y-auto flex-1">
        {items.slice(0, 50).map((node, i) => (
          <div
            key={node.id + i}
            className={`px-3 py-2 cursor-pointer text-sm border-b hover:bg-blue-50 ${selectedId === node.id ? 'bg-blue-100' : ''}`}
            onClick={() => onSelect(node.id)}
          >
            <div className="flex justify-between">
              <span className="text-gray-400 mr-2">#{i + 1}</span>
              <span className="font-mono truncate flex-1">{node.id}</span>
            </div>
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>{node.op}</span>
              <span>{formatNs(node.ns)}</span>
              <span>{node.backend}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

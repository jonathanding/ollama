import { useState } from 'react';
import type { SummaryData } from '../types/trace';
import { formatNs, formatBytes } from '../utils/dataSize';

type Tab = 'ops' | 'copies_size' | 'copies_time';

interface Props {
  data: SummaryData;
  selectedId: string | null;
  onSelect: (id: string) => void;
}

export function HotspotPanel({ data, selectedId, onSelect }: Props) {
  const [tab, setTab] = useState<Tab>('ops');

  const opsRanked = [...data.dag.nodes].sort((a, b) => b.ns - a.ns);
  const copies = data.copy_stats.copies;
  const copiesBySize = [...copies].sort((a, b) => b.est_bytes - a.est_bytes);
  const copiesByTime = [...copies].sort((a, b) => b.total_ns - a.total_ns);

  return (
    <div className="w-72 shrink-0 bg-white border-l flex flex-col h-full">
      <div className="flex items-center justify-between p-2 border-b">
        <span className="font-bold text-sm">Hotspots</span>
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
        {tab === 'ops' && opsRanked.slice(0, 50).map((node, i) => (
          <div
            key={node.id + i}
            className={`px-3 py-2 cursor-pointer text-sm border-b hover:bg-blue-50 ${selectedId === node.id ? 'bg-blue-100' : ''}`}
            onClick={() => onSelect(node.id)}
          >
            <div className="flex justify-between">
              <span className="text-gray-400 mr-2">#{i + 1}</span>
              <span className="font-mono truncate flex-1 text-xs">{node.id}</span>
            </div>
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>{node.op}</span>
              <span>{formatNs(node.ns)}</span>
              <span>{node.backend}</span>
            </div>
          </div>
        ))}
        {tab !== 'ops' && (tab === 'copies_size' ? copiesBySize : copiesByTime).map((c, i) => (
          <div
            key={c.name + i}
            className="px-3 py-2 text-sm border-b"
          >
            <div className="flex justify-between">
              <span className="text-gray-400 mr-2">#{i + 1}</span>
              <span className="font-mono truncate flex-1 text-xs">{c.name}</span>
            </div>
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>{formatBytes(c.est_bytes)}</span>
              <span>{formatNs(c.total_ns)}</span>
              <span>x{c.count}</span>
              <span>{c.backend}</span>
            </div>
          </div>
        ))}
        {tab !== 'ops' && copies.length === 0 && (
          <div className="px-3 py-4 text-sm text-gray-400 text-center">No copy operations found</div>
        )}
      </div>
    </div>
  );
}

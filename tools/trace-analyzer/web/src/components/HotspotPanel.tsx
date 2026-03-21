import { useState } from 'react';
import type { SummaryData } from '../types/trace';
import { formatNs, formatBytes } from '../utils/dataSize';

type Tab = 'ops' | 'copies_size' | 'copies_time';

interface Props {
  data: SummaryData;
  selectedId: string | null;
  onSelect: (id: string) => void;
}

const TAB_LABELS: Record<Tab, string> = {
  ops: 'Top Ops',
  copies_size: 'Copies (size)',
  copies_time: 'Copies (time)',
};

export function HotspotPanel({ data, selectedId, onSelect }: Props) {
  const [tab, setTab] = useState<Tab>('ops');

  const opsRanked = [...data.dag.nodes].sort((a, b) => b.ns - a.ns);
  const maxNs = opsRanked[0]?.ns ?? 1;
  const copies = data.copy_stats.copies;
  const copiesBySize = [...copies].sort((a, b) => b.est_bytes - a.est_bytes);
  const copiesByTime = [...copies].sort((a, b) => b.total_ns - a.total_ns);

  return (
    <div className="w-72 shrink-0 bg-gray-50 border-l flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 bg-indigo-600 text-white">
        <span className="font-semibold text-sm tracking-wide">Hotspots</span>
        <span className="text-indigo-200 text-xs">{opsRanked.length} nodes</span>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 px-3 py-2 bg-white border-b">
        {(['ops', 'copies_size', 'copies_time'] as Tab[]).map(t => (
          <button
            key={t}
            className={`flex-1 px-2 py-1 text-xs rounded-md transition-colors ${tab === t
              ? 'bg-indigo-100 text-indigo-700 font-medium ring-1 ring-indigo-300'
              : 'bg-gray-100 text-gray-500 hover:bg-gray-200'}`}
            onClick={() => setTab(t)}
          >{TAB_LABELS[t]}</button>
        ))}
      </div>

      {/* List */}
      <div className="overflow-y-auto flex-1 mx-2 my-2 bg-white rounded-lg border border-gray-100">
        {tab === 'ops' && opsRanked.slice(0, 50).map((node, i) => {
          const pct = node.ns / maxNs;
          return (
            <div
              key={node.id + i}
              className={`relative px-2.5 py-2 cursor-pointer border-b border-gray-50 hover:bg-indigo-50 transition-colors ${selectedId === node.id ? 'bg-indigo-100' : ''}`}
              onClick={() => onSelect(node.id)}
            >
              {/* Time bar background */}
              <div
                className="absolute inset-y-0 left-0 bg-indigo-50 opacity-50 pointer-events-none"
                style={{ width: `${pct * 100}%` }}
              />
              <div className="relative flex items-baseline gap-2">
                <span className="text-gray-300 w-5 text-right shrink-0 tabular-nums text-[11px]">#{i + 1}</span>
                <span className="font-mono truncate flex-1 text-[11px] text-gray-700">{node.id}</span>
              </div>
              <div className="relative flex items-center gap-3 text-[11px] text-gray-400 mt-0.5 pl-7">
                <span className="font-medium text-gray-600">{node.op}</span>
                <span className="font-mono tabular-nums">{formatNs(node.ns)}</span>
                <span className="bg-gray-100 px-1.5 py-0.5 rounded text-[10px]">{node.backend}</span>
              </div>
            </div>
          );
        })}
        {tab !== 'ops' && (tab === 'copies_size' ? copiesBySize : copiesByTime).map((c, i) => (
          <div
            key={c.name + i}
            className="px-2.5 py-2 border-b border-gray-50"
          >
            <div className="flex items-baseline gap-2">
              <span className="text-gray-300 w-5 text-right shrink-0 tabular-nums text-[11px]">#{i + 1}</span>
              <span className="font-mono truncate flex-1 text-[11px] text-gray-700">{c.name}</span>
            </div>
            <div className="flex items-center gap-3 text-[11px] text-gray-400 mt-0.5 pl-7">
              <span className="font-medium text-gray-600">{formatBytes(c.est_bytes)}</span>
              <span className="font-mono tabular-nums">{formatNs(c.total_ns)}</span>
              <span>x{c.count}</span>
              <span className="bg-gray-100 px-1.5 py-0.5 rounded text-[10px]">{c.backend}</span>
            </div>
          </div>
        ))}
        {tab !== 'ops' && copies.length === 0 && (
          <div className="px-3 py-4 text-xs text-gray-400 text-center">No copy operations found</div>
        )}
      </div>
    </div>
  );
}

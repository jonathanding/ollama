import { useMemo } from 'react';
import { heatmapColor } from '../utils/colorScale';
import { formatNs } from '../utils/dataSize';

const BINS = 10;
const BAR_MAX_H = 56; // px

interface Props {
  durations: number[];
}

export function HeatmapHistogram({ durations }: Props) {
  const data = useMemo(() => {
    if (durations.length === 0) return null;
    const logVals = durations.map(d => Math.log1p(d));
    const logMin = Math.min(...logVals);
    const logMax = Math.max(...logVals);
    const range = logMax - logMin || 1;

    const bins: { count: number; minNs: number; maxNs: number }[] = Array.from(
      { length: BINS }, () => ({ count: 0, minNs: Infinity, maxNs: 0 })
    );
    for (let i = 0; i < durations.length; i++) {
      const idx = Math.min(BINS - 1, Math.floor(((logVals[i] - logMin) / range) * BINS));
      bins[idx].count++;
      bins[idx].minNs = Math.min(bins[idx].minNs, durations[i]);
      bins[idx].maxNs = Math.max(bins[idx].maxNs, durations[i]);
    }
    const maxCount = Math.max(...bins.map(b => b.count), 1);
    return bins.map((bin, i) => ({
      ...bin,
      heightPx: Math.max(2, Math.round((bin.count / maxCount) * BAR_MAX_H)),
      color: heatmapColor((i + 0.5) / BINS),
    }));
  }, [durations]);

  if (!data) return null;

  return (
    <div className="absolute bottom-3 left-3 bg-white/90 backdrop-blur-sm rounded-xl shadow-lg border border-gray-200/80 p-3 z-20 w-60 pointer-events-auto">
      <div className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold mb-2">
        Time Distribution
      </div>
      <div className="flex items-end gap-[3px]" style={{ height: BAR_MAX_H }}>
        {data.map((bin, i) => (
          <div
            key={i}
            className="flex-1 rounded-t-sm cursor-default"
            style={{
              height: bin.heightPx,
              backgroundColor: bin.color,
              border: '1px solid rgba(0,0,0,0.1)',
            }}
            title={`${bin.count} nodes\n${bin.minNs < Infinity ? formatNs(bin.minNs) + ' – ' + formatNs(bin.maxNs) : 'empty'}`}
          />
        ))}
      </div>
      <div className="flex justify-between mt-1.5">
        <span className="text-[10px] text-blue-500 font-medium">fast</span>
        <span className="text-[10px] text-gray-400 tabular-nums">{durations.length} nodes</span>
        <span className="text-[10px] text-orange-500 font-medium">slow</span>
      </div>
    </div>
  );
}

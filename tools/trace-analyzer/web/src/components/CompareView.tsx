import { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import type { CompareData } from '../types/trace';
import { diffColor } from '../utils/colorScale';
import { formatNs } from '../utils/dataSize';

interface Props { data: CompareData; }

type SortKey = 'op' | 'diff';

export function CompareView({ data }: Props) {
  const chartRef = useRef<SVGSVGElement>(null);
  const [sortKey, setSortKey] = useState<SortKey>('diff');

  const sorted = [...data.op_diff].sort((a, b) =>
    sortKey === 'diff' ? Math.abs(b.diff_pct) - Math.abs(a.diff_pct) : a.op.localeCompare(b.op)
  );

  const td = data.timing_diff;
  const cards = [
    { label: 'Total', values: td.total_ms, diff: td.diff_pct.total },
    { label: 'Prefill', values: td.prefill_ms, diff: td.diff_pct.prefill },
    { label: 'Decode avg', values: td.decode_avg_ms, diff: td.diff_pct.decode },
  ];

  useEffect(() => {
    if (!chartRef.current) return;
    const svg = d3.select(chartRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 20, bottom: 60, left: 80 };
    const width = chartRef.current.clientWidth - margin.left - margin.right;
    const height = 250;

    const layers = data.layer_diff;
    const x0 = d3.scaleBand().domain(layers.map(l => l.layer)).range([0, width]).padding(0.3);
    const x1 = d3.scaleBand().domain(data.labels).range([0, x0.bandwidth()]).padding(0.05);
    const maxVal = d3.max(layers, l => d3.max(l.values, v => v.total_ns)) ?? 1;
    const y = d3.scaleLinear().domain([0, maxVal]).range([height, 0]);
    const color = d3.scaleOrdinal(data.labels, ['#4f46e5', '#f97316']);

    const g = svg
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    g.append('g').attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x0)).selectAll('text').attr('transform', 'rotate(-45)').style('text-anchor', 'end');
    g.append('g').call(d3.axisLeft(y).tickFormat(d => formatNs(+d)));

    for (const layer of layers) {
      const lg = g.append('g').attr('transform', `translate(${x0(layer.layer)},0)`);
      for (const val of layer.values) {
        lg.append('rect')
          .attr('x', x1(val.label)!)
          .attr('y', y(val.total_ns))
          .attr('width', x1.bandwidth())
          .attr('height', height - y(val.total_ns))
          .attr('fill', color(val.label))
          .attr('rx', 2)
          .attr('stroke', layer.significant ? '#1e1b4b' : 'none')
          .attr('stroke-width', layer.significant ? 2 : 0);
      }
    }

    const legend = g.append('g').attr('transform', `translate(${width - 120}, 0)`);
    data.labels.forEach((label, i) => {
      legend.append('rect').attr('x', 0).attr('y', i * 20).attr('width', 12).attr('height', 12).attr('fill', color(label)).attr('rx', 2);
      legend.append('text').attr('x', 18).attr('y', i * 20 + 10).text(label).style('font-size', '12px').style('fill', '#4b5563');
    });
  }, [data]);

  return (
    <div className="flex-1 overflow-auto min-h-0 p-4 space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-3 gap-4">
        {cards.map(c => {
          const isSig = Math.abs(c.diff) > 10;
          return (
            <div key={c.label} className="bg-white border border-gray-100 rounded-xl p-4 shadow-sm"
              style={{ borderLeftWidth: 4, borderLeftColor: isSig ? '#ef4444' : '#4f46e5' }}>
              <div className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold">{c.label}</div>
              <div className="flex justify-between mt-2 text-lg font-semibold text-gray-800">
                <span>{c.values[0].toFixed(1)}<span className="text-sm text-gray-400 ml-0.5">ms</span></span>
                <span>{c.values[1].toFixed(1)}<span className="text-sm text-gray-400 ml-0.5">ms</span></span>
              </div>
              <div className="flex justify-between mt-1 text-[11px] text-gray-400">
                <span>{data.labels[0]}</span>
                <span>{data.labels[1]}</span>
              </div>
              <div className={`text-sm mt-2 font-mono tabular-nums ${isSig ? 'font-bold text-red-600' : 'text-gray-500'}`}>
                {c.diff > 0 ? '+' : ''}{c.diff.toFixed(1)}%
              </div>
            </div>
          );
        })}
      </div>

      {/* Op Diff Table */}
      <div className="bg-white rounded-xl border border-gray-100 shadow-sm overflow-hidden">
        <div className="flex items-center gap-2 px-4 py-3 border-b bg-gray-50/50">
          <span className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold">Op Comparison</span>
          <div className="flex gap-0.5 ml-auto bg-gray-100 rounded-md p-0.5">
            <button
              className={`px-2.5 py-1 rounded-md text-xs transition-colors ${sortKey === 'diff' ? 'bg-indigo-600 text-white shadow-sm font-medium' : 'text-gray-500 hover:text-gray-700 hover:bg-gray-200'}`}
              onClick={() => setSortKey('diff')}
            >Sort by diff</button>
            <button
              className={`px-2.5 py-1 rounded-md text-xs transition-colors ${sortKey === 'op' ? 'bg-indigo-600 text-white shadow-sm font-medium' : 'text-gray-500 hover:text-gray-700 hover:bg-gray-200'}`}
              onClick={() => setSortKey('op')}
            >Sort by name</button>
          </div>
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b bg-gray-50/30 text-[11px] uppercase tracking-wider text-gray-400">
              <th className="text-left px-4 py-2 font-semibold">Op</th>
              <th className="text-right px-4 py-2 font-semibold">{data.labels[0]}</th>
              <th className="text-right px-4 py-2 font-semibold">{data.labels[1]}</th>
              <th className="text-right px-4 py-2 font-semibold">Diff</th>
              <th className="text-center px-4 py-2 font-semibold">Sig.</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(o => (
              <tr key={o.op} className="border-b border-gray-50 hover:bg-gray-50/50 transition-colors" style={{ backgroundColor: diffColor(o.diff_pct) }}>
                <td className="px-4 py-2 font-mono text-gray-700">{o.op}</td>
                <td className="px-4 py-2 text-right font-mono tabular-nums text-gray-600">{(o.values[0].total_ns / 1e6).toFixed(1)}</td>
                <td className="px-4 py-2 text-right font-mono tabular-nums text-gray-600">{(o.values[1].total_ns / 1e6).toFixed(1)}</td>
                <td className="px-4 py-2 text-right font-mono tabular-nums font-medium">{o.diff_pct > 0 ? '+' : ''}{o.diff_pct.toFixed(1)}%</td>
                <td className="px-4 py-2 text-center">{o.significant ? <span className="inline-block w-2 h-2 rounded-full bg-red-500" /> : ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Layer Chart */}
      <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-4">
        <h3 className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold mb-3">Layer Comparison</h3>
        <svg ref={chartRef} className="w-full" />
      </div>
    </div>
  );
}

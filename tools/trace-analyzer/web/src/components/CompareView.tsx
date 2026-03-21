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
    const color = d3.scaleOrdinal(data.labels, ['#3b82f6', '#f97316']);

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
          .attr('stroke', layer.significant ? '#000' : 'none')
          .attr('stroke-width', layer.significant ? 2 : 0);
      }
    }

    const legend = g.append('g').attr('transform', `translate(${width - 120}, 0)`);
    data.labels.forEach((label, i) => {
      legend.append('rect').attr('x', 0).attr('y', i * 20).attr('width', 12).attr('height', 12).attr('fill', color(label));
      legend.append('text').attr('x', 18).attr('y', i * 20 + 10).text(label).style('font-size', '12px');
    });
  }, [data]);

  return (
    <div className="p-4 space-y-6">
      <div className="grid grid-cols-3 gap-4">
        {cards.map(c => (
          <div key={c.label} className="border rounded p-3">
            <div className="text-sm text-gray-500">{c.label}</div>
            <div className="flex justify-between mt-1">
              <span>{c.values[0].toFixed(1)}ms</span>
              <span>{c.values[1].toFixed(1)}ms</span>
            </div>
            <div className={`text-sm mt-1 ${Math.abs(c.diff) > 10 ? 'font-bold text-red-600' : 'text-gray-500'}`}>
              {c.diff > 0 ? '+' : ''}{c.diff.toFixed(1)}%
            </div>
          </div>
        ))}
      </div>

      <div>
        <div className="flex gap-2 mb-2">
          <button className={`text-sm px-2 py-1 rounded ${sortKey === 'diff' ? 'bg-gray-800 text-white' : 'bg-gray-200'}`} onClick={() => setSortKey('diff')}>Sort by diff</button>
          <button className={`text-sm px-2 py-1 rounded ${sortKey === 'op' ? 'bg-gray-800 text-white' : 'bg-gray-200'}`} onClick={() => setSortKey('op')}>Sort by name</button>
        </div>
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="border-b">
              <th className="text-left p-2">Op</th>
              <th className="text-right p-2">{data.labels[0]} (ms)</th>
              <th className="text-right p-2">{data.labels[1]} (ms)</th>
              <th className="text-right p-2">Diff</th>
              <th className="text-center p-2">Sig.</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(o => (
              <tr key={o.op} style={{ backgroundColor: diffColor(o.diff_pct) }}>
                <td className="p-2 font-mono">{o.op}</td>
                <td className="p-2 text-right">{(o.values[0].total_ns / 1e6).toFixed(1)}</td>
                <td className="p-2 text-right">{(o.values[1].total_ns / 1e6).toFixed(1)}</td>
                <td className="p-2 text-right">{o.diff_pct > 0 ? '+' : ''}{o.diff_pct.toFixed(1)}%</td>
                <td className="p-2 text-center">{o.significant ? 'Yes' : ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div>
        <h3 className="font-bold mb-2">Layer Comparison</h3>
        <svg ref={chartRef} className="w-full" />
      </div>
    </div>
  );
}

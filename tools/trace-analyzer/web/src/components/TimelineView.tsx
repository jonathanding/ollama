import { useRef, useEffect } from 'react';
import * as d3 from 'd3';
import type { SummaryData, TimelineOp } from '../types/trace';
import { backendColor } from '../utils/colorScale';
import { formatNs } from '../utils/dataSize';

interface Props {
  data: SummaryData;
  onSelectOp: (name: string) => void;
}

export function TimelineView({ data, onSelectOp }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || data.timeline_ops.length === 0) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 30, right: 20, bottom: 30, left: 80 };
    const width = svgRef.current.clientWidth - margin.left - margin.right;
    const passes = data.timing.per_pass;
    const rowHeight = 40;
    const height = passes.length * rowHeight;

    const opsByPass = new Map<number, TimelineOp[]>();
    for (const op of data.timeline_ops) {
      if (!opsByPass.has(op.pass)) opsByPass.set(op.pass, []);
      opsByPass.get(op.pass)!.push(op);
    }

    const passMinT = new Map<number, number>();
    for (const [pid, ops] of opsByPass) {
      passMinT.set(pid, Math.min(...ops.map(o => o.t_start)));
    }
    const maxRelEnd = Math.max(...data.timeline_ops.map(o =>
      o.t_end - (passMinT.get(o.pass) ?? 0)
    ), 1);

    const x = d3.scaleLinear().domain([0, maxRelEnd]).range([0, width]);
    const y = d3.scaleBand()
      .domain(passes.map(p => `Pass ${p.pass}`))
      .range([0, height])
      .padding(0.2);

    const g = svg
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    g.append('g').call(d3.axisLeft(y));
    g.append('g').attr('transform', `translate(0,${height})`).call(d3.axisBottom(x).ticks(10));

    const barHeight = y.bandwidth();

    g.selectAll('.op-rect')
      .data(data.timeline_ops)
      .join('rect')
      .attr('class', 'op-rect')
      .attr('x', d => x(d.t_start - (passMinT.get(d.pass) ?? 0)))
      .attr('y', d => y(`Pass ${d.pass}`) ?? 0)
      .attr('width', d => Math.max(1, x(d.t_end - d.t_start) - x(0)))
      .attr('height', barHeight)
      .attr('fill', d => backendColor(d.backend))
      .attr('stroke', '#fff')
      .attr('stroke-width', 0.5)
      .style('cursor', 'pointer')
      .on('click', (_, d) => onSelectOp(d.name))
      .append('title')
      .text(d => `${d.name}\n${d.op} | ${formatNs(d.t_end - d.t_start)} | ${d.backend}`);

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([1, 100])
      .on('zoom', (event) => {
        const newX = event.transform.rescaleX(x);
        g.selectAll<SVGRectElement, TimelineOp>('.op-rect')
          .attr('x', d => newX(d.t_start - (passMinT.get(d.pass) ?? 0)))
          .attr('width', d => Math.max(1, newX(d.t_end - d.t_start) - newX(0)));
        g.select<SVGGElement>('g:last-of-type').call(d3.axisBottom(newX).ticks(10) as any);
      });
    svg.call(zoom);
  }, [data, onSelectOp]);

  return <svg ref={svgRef} className="w-full h-full min-h-[300px]" />;
}

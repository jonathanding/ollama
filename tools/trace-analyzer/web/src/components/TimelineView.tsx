import { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import type { SummaryData, TimelineOp } from '../types/trace';
import { backendColor, opColor } from '../utils/colorScale';
import { formatNs } from '../utils/dataSize';

const MAX_PASSES = 20;
const DPR = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;

interface Props {
  data: SummaryData;
  onSelectOp: (name: string) => void;
}

interface PreparedOp {
  rx: number;   // normalized x start (0..1)
  rw: number;   // normalized width (0..1)
  row: number;  // row index within visible passes
  color: string;
  op: TimelineOp;
}

export function TimelineView({ data, onSelectOp }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const passes = data.timing.per_pass;
  const totalPasses = passes.length;
  const [startPass, setStartPass] = useState(0);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);
  const [colorBy, setColorBy] = useState<'backend' | 'op'>('backend');

  // Build legend entries from visible ops
  const legend = useMemo(() => {
    const ops = data.timeline_ops;
    const seen = new Map<string, string>(); // label → color
    for (const op of ops) {
      const label = colorBy === 'op' ? op.op : op.backend;
      if (!seen.has(label)) {
        seen.set(label, colorBy === 'op' ? opColor(op.op) : backendColor(op.backend));
      }
    }
    // Sort by label
    return [...seen.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }, [data.timeline_ops, colorBy]);

  const zoomRef = useRef({ tx: 0, kx: 1 });
  const preparedRef = useRef<PreparedOp[]>([]);
  const layoutRef = useRef({
    margin: { top: 30, right: 20, bottom: 30, left: 80 },
    width: 0, height: 0, rowHeight: 40, barHeight: 32, maxRelEnd: 1,
  });

  const visiblePasses = passes.slice(startPass, startPass + MAX_PASSES);
  const visiblePassIds = new Set(visiblePasses.map(p => p.pass));
  const passToRow = new Map<number, number>();
  visiblePasses.forEach((p, i) => passToRow.set(p.pass, i));

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { margin, width, height, rowHeight, barHeight, maxRelEnd } = layoutRef.current;
    const { tx, kx } = zoomRef.current;
    const prepared = preparedRef.current;

    const totalW = margin.left + width + margin.right;
    const totalH = margin.top + height + margin.bottom;

    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, totalW, totalH);

    // Y-axis labels
    ctx.fillStyle = '#374151';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < visiblePasses.length; i++) {
      ctx.fillText(
        `Pass ${visiblePasses[i].pass}`,
        margin.left - 8,
        margin.top + i * rowHeight + rowHeight / 2,
      );
    }

    // Alternating row backgrounds
    ctx.fillStyle = '#f9fafb';
    for (let i = 0; i < visiblePasses.length; i += 2) {
      ctx.fillRect(margin.left, margin.top + i * rowHeight, width, rowHeight);
    }

    // Clip chart area
    ctx.save();
    ctx.beginPath();
    ctx.rect(margin.left, margin.top, width, height);
    ctx.clip();

    // Draw op rects — rx/rw are 0..1 normalized, scale by width
    for (const p of prepared) {
      const px = margin.left + p.rx * width * kx + tx;
      const pw = p.rw * width * kx;
      if (px + pw < margin.left || px > margin.left + width) continue;
      const py = margin.top + p.row * rowHeight + (rowHeight - barHeight) / 2;
      ctx.fillStyle = p.color;
      ctx.fillRect(px, py, Math.max(0.5, pw), barHeight);
    }

    ctx.restore();

    // X-axis ticks
    ctx.fillStyle = '#374151';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.strokeStyle = '#d1d5db';
    ctx.lineWidth = 1;
    const tickCount = 10;
    for (let i = 0; i <= tickCount; i++) {
      const screenX = margin.left + (i / tickCount) * width;
      // Invert zoom transform: screenX = margin.left + normalizedVal * width * kx + tx
      // normalizedVal = (screenX - margin.left - tx) / (width * kx)
      const normalizedVal = (screenX - margin.left - tx) / (width * kx);
      const domainVal = normalizedVal * maxRelEnd;
      if (domainVal < 0) continue;
      ctx.beginPath();
      ctx.moveTo(screenX, margin.top + height);
      ctx.lineTo(screenX, margin.top + height + 5);
      ctx.stroke();
      ctx.fillText(formatNs(domainVal), screenX, margin.top + height + 7);
    }

    // Border
    ctx.strokeStyle = '#d1d5db';
    ctx.lineWidth = 1;
    ctx.strokeRect(margin.left, margin.top, width, height);
  }, [visiblePasses]);

  // Prepare data and initial draw
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || data.timeline_ops.length === 0) return;

    const ops = data.timeline_ops.filter(o => visiblePassIds.has(o.pass));
    if (ops.length === 0) return;

    const margin = { top: 30, right: 20, bottom: 30, left: 80 };
    const containerWidth = container.clientWidth;
    const width = containerWidth - margin.left - margin.right;
    const rowHeight = 40;
    const barHeight = 32;
    const height = visiblePasses.length * rowHeight;

    canvas.width = (width + margin.left + margin.right) * DPR;
    canvas.height = (height + margin.top + margin.bottom) * DPR;
    canvas.style.width = `${width + margin.left + margin.right}px`;
    canvas.style.height = `${height + margin.top + margin.bottom}px`;

    const opsByPass = new Map<number, TimelineOp[]>();
    for (const op of ops) {
      if (!opsByPass.has(op.pass)) opsByPass.set(op.pass, []);
      opsByPass.get(op.pass)!.push(op);
    }
    const passMinT = new Map<number, number>();
    for (const [pid, pops] of opsByPass) {
      passMinT.set(pid, pops.reduce((m, o) => Math.min(m, o.t_start), Infinity));
    }
    const maxRelEnd = ops.reduce((m, o) =>
      Math.max(m, o.t_end - (passMinT.get(o.pass) ?? 0)), 1);

    const prepared: PreparedOp[] = ops.map(op => {
      const relStart = op.t_start - (passMinT.get(op.pass) ?? 0);
      const duration = op.t_end - op.t_start;
      return {
        rx: relStart / maxRelEnd,
        rw: duration / maxRelEnd,
        row: passToRow.get(op.pass) ?? 0,
        color: colorBy === 'op' ? opColor(op.op) : backendColor(op.backend),
        op,
      };
    });

    preparedRef.current = prepared;
    layoutRef.current = { margin, width, height, rowHeight, barHeight, maxRelEnd };
    zoomRef.current = { tx: 0, kx: 1 };

    draw();
  }, [data, startPass, colorBy]);

  // Zoom via wheel, pan via shift+wheel or horizontal scroll
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      const z = zoomRef.current;
      const { width } = layoutRef.current;
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left - layoutRef.current.margin.left;

      if (e.ctrlKey || e.metaKey) {
        const factor = e.deltaY < 0 ? 1.1 : 0.9;
        const newK = Math.max(1, Math.min(500, z.kx * factor));
        const newTx = mouseX - (mouseX - z.tx) * (newK / z.kx);
        z.kx = newK;
        z.tx = Math.min(0, Math.max(width - width * newK, newTx));
      } else {
        const dx = e.deltaX || (e.shiftKey ? e.deltaY : 0);
        z.tx = Math.min(0, Math.max(width - width * z.kx, z.tx - dx));
      }
      draw();
    };

    canvas.addEventListener('wheel', handleWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', handleWheel);
  }, [draw]);

  // Click & hover for tooltip
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const hitTest = (clientX: number, clientY: number): PreparedOp | null => {
      const rect = canvas.getBoundingClientRect();
      const mx = clientX - rect.left;
      const my = clientY - rect.top;
      const { margin, width, rowHeight, barHeight } = layoutRef.current;
      const { tx, kx } = zoomRef.current;

      for (const p of preparedRef.current) {
        const px = margin.left + p.rx * width * kx + tx;
        const pw = Math.max(3, p.rw * width * kx);
        const py = margin.top + p.row * rowHeight + (rowHeight - barHeight) / 2;
        if (mx >= px && mx <= px + pw && my >= py && my <= py + barHeight) {
          return p;
        }
      }
      return null;
    };

    const handleClick = (e: MouseEvent) => {
      const hit = hitTest(e.clientX, e.clientY);
      if (hit) onSelectOp(hit.op.name);
    };

    const handleMove = (e: MouseEvent) => {
      const hit = hitTest(e.clientX, e.clientY);
      if (hit) {
        canvas.style.cursor = 'pointer';
        const rect = canvas.getBoundingClientRect();
        setTooltip({
          x: e.clientX - rect.left + 12,
          y: e.clientY - rect.top - 10,
          text: `${hit.op.name}\n${hit.op.op} | ${formatNs(hit.op.t_end - hit.op.t_start)} | ${hit.op.backend}`,
        });
      } else {
        canvas.style.cursor = 'default';
        setTooltip(null);
      }
    };

    const handleLeave = () => setTooltip(null);

    canvas.addEventListener('click', handleClick);
    canvas.addEventListener('mousemove', handleMove);
    canvas.addEventListener('mouseleave', handleLeave);
    return () => {
      canvas.removeEventListener('click', handleClick);
      canvas.removeEventListener('mousemove', handleMove);
      canvas.removeEventListener('mouseleave', handleLeave);
    };
  }, [onSelectOp]);

  // Drag to pan
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let dragging = false;
    let lastX = 0;

    const handleDown = (e: MouseEvent) => {
      dragging = true;
      lastX = e.clientX;
      canvas.style.cursor = 'grabbing';
    };
    const handleMove = (e: MouseEvent) => {
      if (!dragging) return;
      const dx = e.clientX - lastX;
      lastX = e.clientX;
      const z = zoomRef.current;
      const { width } = layoutRef.current;
      z.tx = Math.min(0, Math.max(width - width * z.kx, z.tx + dx));
      draw();
    };
    const handleUp = () => {
      dragging = false;
      canvas.style.cursor = 'default';
    };

    canvas.addEventListener('mousedown', handleDown);
    window.addEventListener('mousemove', handleMove);
    window.addEventListener('mouseup', handleUp);
    return () => {
      canvas.removeEventListener('mousedown', handleDown);
      window.removeEventListener('mousemove', handleMove);
      window.removeEventListener('mouseup', handleUp);
    };
  }, [draw]);

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {totalPasses > MAX_PASSES && (
        <div className="flex items-center gap-3 p-2 border-b bg-white shrink-0 text-sm">
          <button
            className="px-2 py-1 rounded bg-gray-200 hover:bg-gray-300 disabled:opacity-30"
            disabled={startPass === 0}
            onClick={() => setStartPass(Math.max(0, startPass - MAX_PASSES))}
          >Prev</button>
          <span className="text-gray-600">
            Pass {startPass}–{Math.min(startPass + MAX_PASSES - 1, totalPasses - 1)} of {totalPasses}
          </span>
          <button
            className="px-2 py-1 rounded bg-gray-200 hover:bg-gray-300 disabled:opacity-30"
            disabled={startPass + MAX_PASSES >= totalPasses}
            onClick={() => setStartPass(Math.min(totalPasses - MAX_PASSES, startPass + MAX_PASSES))}
          >Next</button>
        </div>
      )}
      <div className="flex items-center gap-3 px-2 py-1 shrink-0 text-xs text-gray-500">
        <span>Color:</span>
        <button
          className={`px-2 py-0.5 rounded ${colorBy === 'backend' ? 'bg-gray-800 text-white' : 'bg-gray-200'}`}
          onClick={() => setColorBy('backend')}
        >Backend</button>
        <button
          className={`px-2 py-0.5 rounded ${colorBy === 'op' ? 'bg-gray-800 text-white' : 'bg-gray-200'}`}
          onClick={() => setColorBy('op')}
        >Op Type</button>
        <div className="w-px h-4 bg-gray-300 mx-1" />
        {legend.map(([label, color]) => (
          <span key={label} className="inline-flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded-sm shrink-0" style={{ backgroundColor: color }} />
            <span className="text-gray-600">{label}</span>
          </span>
        ))}
        <span className="text-gray-400 ml-auto">Ctrl+Scroll to zoom · Drag or Shift+Scroll to pan</span>
      </div>
      <div ref={containerRef} className="flex-1 overflow-auto min-h-0 relative">
        <canvas ref={canvasRef} className="min-h-[300px]" />
        {tooltip && (
          <div
            className="absolute pointer-events-none bg-gray-800 text-white text-xs px-2 py-1 rounded shadow-lg whitespace-pre z-50"
            style={{ left: tooltip.x, top: tooltip.y }}
          >
            {tooltip.text}
          </div>
        )}
      </div>
    </div>
  );
}

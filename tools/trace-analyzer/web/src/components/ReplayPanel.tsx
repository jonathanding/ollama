// src/components/ReplayPanel.tsx
import { useState, useRef, useEffect, useCallback } from 'react';
import type { SummaryData, TimelineOp, DagNode } from '../types/trace';
import { ReplayController } from '../utils/ReplayController';
import { heatmapColor } from '../utils/colorScale';
import { formatNs } from '../utils/dataSize';

export type ExpandMode = 'keep' | 'auto';

export interface ReplayState {
  currentNodeId: string;
  visitedNodeIds: Set<string>;
}

interface Props {
  data: SummaryData;
  onReplayState: (state: ReplayState | null) => void;
  onStop: () => void;
  expandMode: ExpandMode;
  onExpandModeChange: (mode: ExpandMode) => void;
}

const SPEED_PRESETS = [1, 2, 5, 10, 20];

export function ReplayPanel({ data, onReplayState, onStop, expandMode, onExpandModeChange }: Props) {
  const [selectedPass, setSelectedPass] = useState(0);
  const [currentStep, setCurrentStep] = useState<TimelineOp | null>(null);
  const [currentIndex, setCurrentIndex] = useState(-1);
  const [log, setLog] = useState<TimelineOp[]>([]);
  const [speed, setSpeed] = useState(1);
  const [playing, setPlaying] = useState(false);
  const ctrlRef = useRef<ReplayController | null>(null);
  const visitedRef = useRef<Set<string>>(new Set());
  const logRef = useRef<HTMLDivElement>(null);

  const dagNodeIds = new Set(data.dag.nodes.map(n => n.id));
  const passes = data.timing.per_pass;

  const steps = data.timeline_ops
    .filter(op => op.pass === passes[selectedPass]?.pass && dagNodeIds.has(op.name))
    .sort((a, b) => a.seq - b.seq);

  const durations = steps.map(s => s.t_end - s.t_start);
  const minDur = Math.min(...durations, 0);
  const maxDur = Math.max(...durations, 1);

  const handleStep = useCallback((step: TimelineOp, index: number) => {
    visitedRef.current.add(step.name);
    setCurrentStep(step);
    setCurrentIndex(index);
    setLog(prev => [step, ...prev].slice(0, 50));
    onReplayState({
      currentNodeId: step.name,
      visitedNodeIds: new Set(visitedRef.current),
    });
    setTimeout(() => logRef.current?.scrollTo({ top: 0, behavior: 'smooth' }), 0);
  }, [onReplayState]);

  const handleFinish = useCallback(() => {
    setPlaying(false);
  }, []);

  useEffect(() => {
    ctrlRef.current?.dispose();
    const ctrl = new ReplayController(steps, { onStep: handleStep, onFinish: handleFinish });
    ctrl.setSpeed(speed);
    ctrlRef.current = ctrl;
    setCurrentStep(null);
    setCurrentIndex(-1);
    setLog([]);
    setPlaying(false);
    visitedRef.current = new Set();
    onReplayState(null);
    return () => ctrl.dispose();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- intentionally recreate only on pass/data change; callbacks are stable refs
  }, [selectedPass, data]);

  const handlePlay = () => {
    const ctrl = ctrlRef.current;
    if (!ctrl) return;
    if (ctrl.state === 'paused') {
      ctrl.resume();
    } else {
      visitedRef.current = new Set();
      setLog([]);
      ctrl.play();
    }
    setPlaying(true);
  };

  const handlePause = () => {
    ctrlRef.current?.pause();
    setPlaying(false);
  };

  const handleStopClick = () => {
    ctrlRef.current?.stop();
    setPlaying(false);
    setCurrentStep(null);
    setCurrentIndex(-1);
    setLog([]);
    visitedRef.current = new Set();
    onReplayState(null);
    onStop();
  };

  const handleSpeedChange = (s: number) => {
    setSpeed(s);
    ctrlRef.current?.setSpeed(s);
  };

  const handleSeek = (index: number) => {
    const newVisited = new Set<string>();
    for (let i = 0; i <= index; i++) {
      newVisited.add(steps[i].name);
    }
    visitedRef.current = newVisited;
    ctrlRef.current?.seekTo(index);
  };

  const passLabel = (p: { pass: number; n_tokens: number; n_ops: number }) =>
    `Pass ${p.pass} (${p.n_tokens > 1 ? 'prefill' : 'decode'}, ${p.n_ops} ops)`;

  const dagNode: DagNode | undefined = currentStep
    ? data.dag.nodes.find(n => n.id === currentStep.name)
    : undefined;

  const duration = currentStep ? currentStep.t_end - currentStep.t_start : 0;
  const heatRatio = maxDur > minDur ? (duration - minDur) / (maxDur - minDur) : 0;

  return (
    <div className="w-72 shrink-0 bg-gray-50 border-l flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 bg-indigo-600 text-white">
        <span className="font-semibold text-sm tracking-wide">Replay</span>
        <button
          className="text-indigo-200 hover:text-white text-lg leading-none"
          onClick={handleStopClick}
          title="Close replay"
        >&times;</button>
      </div>

      {/* Pass selector */}
      <div className="px-3 py-2 bg-white border-b">
        <label className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold">Pass</label>
        <select
          className="w-full border border-gray-200 rounded-md px-2 py-1.5 text-xs mt-1 bg-white focus:border-indigo-400 focus:ring-1 focus:ring-indigo-200 outline-none"
          value={selectedPass}
          onChange={e => { setSelectedPass(Number(e.target.value)); }}
        >
          {passes.map((p, i) => (
            <option key={p.pass} value={i}>{passLabel(p)}</option>
          ))}
        </select>
      </div>

      {/* Layer mode */}
      <div className="px-3 py-2 bg-white border-b">
        <label className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold">Layer Folding</label>
        <div className="flex gap-1 mt-1">
          <button
            className={`flex-1 px-2 py-1 text-xs rounded-md transition-colors ${expandMode === 'keep'
              ? 'bg-indigo-100 text-indigo-700 font-medium ring-1 ring-indigo-300'
              : 'bg-gray-100 text-gray-500 hover:bg-gray-200'}`}
            onClick={() => onExpandModeChange('keep')}
            title="Keep current layer expand/collapse state during replay"
          >Manual</button>
          <button
            className={`flex-1 px-2 py-1 text-xs rounded-md transition-colors ${expandMode === 'auto'
              ? 'bg-indigo-100 text-indigo-700 font-medium ring-1 ring-indigo-300'
              : 'bg-gray-100 text-gray-500 hover:bg-gray-200'}`}
            onClick={() => onExpandModeChange('auto')}
            title="Auto-expand the active layer and collapse others"
          >Auto Expand</button>
        </div>
      </div>

      {/* Transport controls */}
      <div className="px-3 py-2 bg-white border-b space-y-2">
        <div className="flex items-center gap-1.5">
          {playing ? (
            <button className="px-4 py-1.5 text-xs rounded-md bg-amber-500 text-white hover:bg-amber-600 font-medium shadow-sm transition-colors"
              onClick={handlePause}>Pause</button>
          ) : (
            <button className="px-4 py-1.5 text-xs rounded-md bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-30 font-medium shadow-sm transition-colors"
              onClick={handlePlay} disabled={steps.length === 0}>Play</button>
          )}
          <button className="px-3 py-1.5 text-xs rounded-md bg-gray-200 text-gray-600 hover:bg-gray-300 transition-colors"
            onClick={handleStopClick}>Stop</button>
          <span className="text-xs text-gray-400 ml-auto font-mono tabular-nums">
            {currentIndex >= 0 ? currentIndex + 1 : 0}<span className="text-gray-300">/</span>{steps.length}
          </span>
        </div>

        {/* Speed */}
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold shrink-0">Speed</span>
          <div className="flex gap-0.5 flex-1">
            {SPEED_PRESETS.map(s => (
              <button key={s}
                className={`flex-1 py-0.5 text-[11px] rounded transition-colors ${speed === s
                  ? 'bg-indigo-600 text-white font-medium'
                  : 'bg-gray-100 text-gray-500 hover:bg-gray-200'}`}
                onClick={() => handleSpeedChange(s)}>{s}x</button>
            ))}
          </div>
        </div>

        {/* Progress */}
        <input type="range" min={0} max={Math.max(0, steps.length - 1)}
          value={Math.max(0, currentIndex)}
          onChange={e => handleSeek(Number(e.target.value))}
          className="w-full h-1 cursor-pointer accent-indigo-600" />
      </div>

      {/* Current op card */}
      {currentStep && (
        <div className="mx-3 mt-2 mb-1 p-2.5 bg-white rounded-lg shadow-sm border border-gray-100 text-xs space-y-1"
          style={{ borderLeftWidth: 4, borderLeftColor: heatmapColor(heatRatio) }}>
          <div className="font-bold text-sm text-gray-800">{currentStep.op}</div>
          <div className="font-mono text-gray-500 truncate text-[11px]">{currentStep.name}</div>
          <div className="flex justify-between text-gray-400 pt-0.5">
            <span className="bg-gray-100 px-1.5 py-0.5 rounded text-[10px]">{currentStep.backend}</span>
            <span className="font-mono tabular-nums">{formatNs(duration)}</span>
          </div>
          {dagNode && (
            <div className="text-gray-400 font-mono text-[10px] pt-0.5">
              [{dagNode.shape.join(', ')}] {dagNode.dtype}
            </div>
          )}
        </div>
      )}

      {/* Log header */}
      <div className="px-3 pt-2 pb-1">
        <span className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold">Op Log</span>
      </div>

      {/* Log queue */}
      <div ref={logRef} className="overflow-y-auto flex-1 text-xs mx-2 mb-2 bg-white rounded-lg border border-gray-100">
        {log.map((op, i) => {
          const isActive = i === 0 && currentStep?.seq === op.seq;
          return (
            <div key={`${op.seq}-${i}`}
              className={`px-2 py-1 border-b border-gray-50 cursor-pointer hover:bg-indigo-50 flex gap-2 transition-colors ${isActive ? 'bg-indigo-100' : ''}`}
              onClick={() => handleSeek(steps.findIndex(s => s.seq === op.seq))}>
              <span className="text-gray-300 w-6 text-right shrink-0 tabular-nums">#{op.seq}</span>
              <span className="font-medium w-16 truncate shrink-0 text-gray-700">{op.op}</span>
              <span className="font-mono truncate flex-1 text-gray-400">{op.name}</span>
              <span className="text-gray-300 shrink-0 tabular-nums">{formatNs(op.t_end - op.t_start)}</span>
            </div>
          );
        })}
        {steps.length === 0 && (
          <div className="px-3 py-4 text-gray-400 text-center">No ops in this pass</div>
        )}
      </div>
    </div>
  );
}

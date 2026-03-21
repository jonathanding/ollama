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
    <div className="w-72 shrink-0 bg-white border-l flex flex-col h-full">
      <div className="flex items-center justify-between p-2 border-b">
        <span className="font-bold text-sm">Replay</span>
        <button
          className="text-gray-400 hover:text-gray-600 text-lg leading-none"
          onClick={handleStopClick}
          title="Close replay"
        >&times;</button>
      </div>

      <div className="p-2 border-b space-y-2">
        <select
          className="w-full border rounded px-2 py-1 text-xs"
          value={selectedPass}
          onChange={e => { setSelectedPass(Number(e.target.value)); }}
        >
          {passes.map((p, i) => (
            <option key={p.pass} value={i}>{passLabel(p)}</option>
          ))}
        </select>

        <div className="flex gap-2 text-xs">
          <label className="flex items-center gap-1 cursor-pointer">
            <input type="radio" name="expandMode" checked={expandMode === 'keep'}
              onChange={() => onExpandModeChange('keep')} className="w-3 h-3" />
            Keep Current
          </label>
          <label className="flex items-center gap-1 cursor-pointer">
            <input type="radio" name="expandMode" checked={expandMode === 'auto'}
              onChange={() => onExpandModeChange('auto')} className="w-3 h-3" />
            Auto Expand
          </label>
        </div>
      </div>

      <div className="p-2 border-b space-y-2">
        <div className="flex items-center gap-1">
          {playing ? (
            <button className="px-3 py-1 text-xs rounded bg-gray-800 text-white hover:bg-gray-700"
              onClick={handlePause}>Pause</button>
          ) : (
            <button className="px-3 py-1 text-xs rounded bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-30"
              onClick={handlePlay} disabled={steps.length === 0}>Play</button>
          )}
          <button className="px-3 py-1 text-xs rounded bg-gray-200 hover:bg-gray-300"
            onClick={handleStopClick}>Stop</button>
          <span className="text-xs text-gray-400 ml-auto">
            {currentIndex >= 0 ? currentIndex + 1 : 0}/{steps.length}
          </span>
        </div>

        <div className="flex gap-1">
          {SPEED_PRESETS.map(s => (
            <button key={s}
              className={`flex-1 py-0.5 text-xs rounded ${speed === s ? 'bg-gray-800 text-white' : 'bg-gray-200'}`}
              onClick={() => handleSpeedChange(s)}>{s}x</button>
          ))}
        </div>

        <input type="range" min={0} max={Math.max(0, steps.length - 1)}
          value={Math.max(0, currentIndex)}
          onChange={e => handleSeek(Number(e.target.value))}
          className="w-full h-1 cursor-pointer" />
      </div>

      {currentStep && (
        <div className="m-2 p-2 border rounded text-xs space-y-1"
          style={{ borderLeftWidth: 4, borderLeftColor: heatmapColor(heatRatio) }}>
          <div className="font-bold text-base">{currentStep.op}</div>
          <div className="font-mono text-gray-600 truncate">{currentStep.name}</div>
          <div className="flex justify-between text-gray-500">
            <span>{currentStep.backend}</span>
            <span>{formatNs(duration)}</span>
          </div>
          {dagNode && (
            <div className="text-gray-400">
              [{dagNode.shape.join(', ')}] {dagNode.dtype}
            </div>
          )}
        </div>
      )}

      <div ref={logRef} className="overflow-y-auto flex-1 text-xs">
        {log.map((op, i) => {
          const isActive = i === 0 && currentStep?.seq === op.seq;
          return (
            <div key={`${op.seq}-${i}`}
              className={`px-2 py-1 border-b cursor-pointer hover:bg-blue-50 flex gap-2 ${isActive ? 'bg-blue-100' : ''}`}
              onClick={() => handleSeek(steps.findIndex(s => s.seq === op.seq))}>
              <span className="text-gray-400 w-6 text-right shrink-0">#{op.seq}</span>
              <span className="font-medium w-16 truncate shrink-0">{op.op}</span>
              <span className="font-mono truncate flex-1 text-gray-500">{op.name}</span>
              <span className="text-gray-400 shrink-0">{formatNs(op.t_end - op.t_start)}</span>
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

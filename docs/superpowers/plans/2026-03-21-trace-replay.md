# Trace Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Animate computation graph execution on the DAG view, stepping through ops in execution order to visualize data flow — pure frontend, no backend changes.

**Architecture:** ReplayController (pure TS state machine) drives step progression via setInterval. ReplayPanel (React sidebar) replaces HotspotPanel when active, owning the controller and exposing controls/log. DagView receives replay state via props and applies Cytoscape classes for visited/current/edge highlighting.

**Tech Stack:** React 18, TypeScript, Cytoscape.js (existing)

---

## File Structure

```
web/src/
├── utils/
│   └── ReplayController.ts     # Pure TS class, no React — state machine + timer
├── components/
│   ├── ReplayPanel.tsx          # Controls + current op card + log queue
│   └── DagView.tsx              # +replay useEffect, +3 Cytoscape style selectors
└── App.tsx                      # +replayActive state, conditional HotspotPanel/ReplayPanel
```

New dev dependency: `vitest` (for unit tests). No Python backend changes. No changes to summary.json schema.

---

### Task 1: ReplayController — Pure TS State Machine

**Files:**
- Create: `web/src/utils/ReplayController.ts`
- Create: `web/src/__tests__/ReplayController.test.ts`

This is the core engine. No React dependency — just a class with play/pause/stop/seek and
a timer that calls `onStep` callbacks.

- [ ] **Step 0: Install vitest**

```bash
cd web && npm install -D vitest
```

- [ ] **Step 1: Write the failing test for basic lifecycle**

```ts
// web/src/__tests__/ReplayController.test.ts
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ReplayController } from '../utils/ReplayController';
import type { TimelineOp } from '../types/trace';

function makeOps(n: number, pass = 0): TimelineOp[] {
  return Array.from({ length: n }, (_, i) => ({
    pass,
    seq: i,
    name: `op_${i}`,
    op: 'MUL_MAT',
    backend: 'CUDA0',
    t_start: i * 1000,
    t_end: (i + 1) * 1000,
  }));
}

describe('ReplayController', () => {
  beforeEach(() => { vi.useFakeTimers(); });
  afterEach(() => { vi.restoreAllTimers(); });

  it('starts in idle state', () => {
    const ctrl = new ReplayController(makeOps(5), { onStep: vi.fn(), onFinish: vi.fn() });
    expect(ctrl.state).toBe('idle');
    expect(ctrl.currentIndex).toBe(-1);
  });

  it('play transitions to playing and calls onStep', () => {
    const onStep = vi.fn();
    const onFinish = vi.fn();
    const ops = makeOps(3);
    const ctrl = new ReplayController(ops, { onStep, onFinish });

    ctrl.play();
    expect(ctrl.state).toBe('playing');
    // First step fires immediately
    expect(onStep).toHaveBeenCalledWith(ops[0], 0);

    vi.advanceTimersByTime(80); // BASE_INTERVAL
    expect(onStep).toHaveBeenCalledWith(ops[1], 1);

    vi.advanceTimersByTime(80);
    expect(onStep).toHaveBeenCalledWith(ops[2], 2);

    vi.advanceTimersByTime(80);
    expect(onFinish).toHaveBeenCalled();
    expect(ctrl.state).toBe('idle');
  });

  it('pause and resume work', () => {
    const onStep = vi.fn();
    const ctrl = new ReplayController(makeOps(10), { onStep, onFinish: vi.fn() });

    ctrl.play();
    vi.advanceTimersByTime(80);
    ctrl.pause();
    expect(ctrl.state).toBe('paused');

    const callCount = onStep.mock.calls.length;
    vi.advanceTimersByTime(400);
    expect(onStep.mock.calls.length).toBe(callCount); // no new calls

    ctrl.resume();
    expect(ctrl.state).toBe('playing');
    vi.advanceTimersByTime(80);
    expect(onStep.mock.calls.length).toBe(callCount + 1);
  });

  it('stop resets to idle', () => {
    const onStep = vi.fn();
    const ctrl = new ReplayController(makeOps(10), { onStep, onFinish: vi.fn() });
    ctrl.play();
    vi.advanceTimersByTime(160);
    ctrl.stop();
    expect(ctrl.state).toBe('idle');
    expect(ctrl.currentIndex).toBe(-1);
  });

  it('setSpeed changes interval', () => {
    const onStep = vi.fn();
    const ctrl = new ReplayController(makeOps(20), { onStep, onFinish: vi.fn() });
    ctrl.setSpeed(2);
    ctrl.play();
    // At 2x, interval = 80/2 = 40ms
    vi.advanceTimersByTime(40);
    expect(onStep).toHaveBeenCalledTimes(2); // initial + 1 tick
  });

  it('seekTo updates index immediately', () => {
    const onStep = vi.fn();
    const ctrl = new ReplayController(makeOps(10), { onStep, onFinish: vi.fn() });
    ctrl.play();
    ctrl.pause();
    onStep.mockClear();

    ctrl.seekTo(5);
    expect(ctrl.currentIndex).toBe(5);
    expect(onStep).toHaveBeenCalledWith(expect.objectContaining({ seq: 5 }), 5);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run src/__tests__/ReplayController.test.ts`
Expected: FAIL — module not found

- [ ] **Step 3: Implement ReplayController**

```ts
// web/src/utils/ReplayController.ts
import type { TimelineOp } from '../types/trace';

export interface ReplayCallbacks {
  onStep: (step: TimelineOp, index: number) => void;
  onFinish: () => void;
}

const BASE_INTERVAL = 80; // ms

export class ReplayController {
  state: 'idle' | 'playing' | 'paused' = 'idle';
  steps: TimelineOp[];
  currentIndex = -1;
  speed = 1;

  private callbacks: ReplayCallbacks;
  private timerId: ReturnType<typeof setInterval> | null = null;

  constructor(steps: TimelineOp[], callbacks: ReplayCallbacks) {
    this.steps = steps;
    this.callbacks = callbacks;
  }

  play(): void {
    if (this.state === 'playing' || this.steps.length === 0) return;
    this.state = 'playing';
    this.currentIndex = 0;
    this.callbacks.onStep(this.steps[0], 0);
    this.startTimer();
  }

  pause(): void {
    if (this.state !== 'playing') return;
    this.state = 'paused';
    this.clearTimer();
  }

  resume(): void {
    if (this.state !== 'paused') return;
    this.state = 'playing';
    this.startTimer();
  }

  stop(): void {
    this.state = 'idle';
    this.currentIndex = -1;
    this.clearTimer();
  }

  setSpeed(n: number): void {
    this.speed = n;
    if (this.state === 'playing') {
      this.clearTimer();
      this.startTimer();
    }
  }

  seekTo(index: number): void {
    if (index < 0 || index >= this.steps.length) return;
    this.currentIndex = index;
    this.callbacks.onStep(this.steps[index], index);
  }

  dispose(): void {
    this.clearTimer();
  }

  private startTimer(): void {
    this.clearTimer();
    this.timerId = setInterval(() => this.advance(), BASE_INTERVAL / this.speed);
  }

  private clearTimer(): void {
    if (this.timerId !== null) {
      clearInterval(this.timerId);
      this.timerId = null;
    }
  }

  private advance(): void {
    const next = this.currentIndex + 1;
    if (next >= this.steps.length) {
      this.clearTimer();
      this.state = 'idle';
      this.callbacks.onFinish();
      return;
    }
    this.currentIndex = next;
    this.callbacks.onStep(this.steps[next], next);
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd web && npx vitest run src/__tests__/ReplayController.test.ts`
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add web/src/utils/ReplayController.ts web/src/__tests__/ReplayController.test.ts
git commit -m "tools: add ReplayController pure TS state machine"
```

---

### Task 2: ReplayPanel — Controls + Detail Card + Log Queue

**Files:**
- Create: `web/src/components/ReplayPanel.tsx`

ReplayPanel replaces HotspotPanel in the right sidebar when replay is active. It owns a
ReplayController instance and communicates replay state up to App via callbacks.

Three sections stacked vertically:
1. **Top: Controls** — pass selector, expand mode radio, transport buttons, speed presets, progress bar
2. **Middle: Current Op Card** — op type, tensor name, backend, duration, shape/dtype
3. **Bottom: Op Log Queue** — scrollable list of visited ops, click to seek

- [ ] **Step 1: Create ReplayPanel component**

```tsx
// web/src/components/ReplayPanel.tsx
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

  // Build steps for selected pass, filtered to nodes that exist in DAG
  const dagNodeIds = new Set(data.dag.nodes.map(n => n.id));
  const passes = data.timing.per_pass;

  const steps = data.timeline_ops
    .filter(op => op.pass === passes[selectedPass]?.pass && dagNodeIds.has(op.name))
    .sort((a, b) => a.seq - b.seq);

  // Compute duration range for heatmap
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
    // Auto-scroll log
    setTimeout(() => logRef.current?.scrollTo({ top: 0, behavior: 'smooth' }), 0);
  }, [onReplayState]);

  const handleFinish = useCallback(() => {
    setPlaying(false);
  }, []);

  // Recreate controller when steps change
  useEffect(() => {
    ctrlRef.current?.dispose();
    const ctrl = new ReplayController(steps, { onStep: handleStep, onFinish: handleFinish });
    ctrl.setSpeed(speed);
    ctrlRef.current = ctrl;
    // Reset state
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
    // Rebuild visited set up to index
    const newVisited = new Set<string>();
    for (let i = 0; i <= index; i++) {
      newVisited.add(steps[i].name);
    }
    visitedRef.current = newVisited;
    ctrlRef.current?.seekTo(index);
  };

  const passLabel = (p: { pass: number; n_tokens: number; n_ops: number }) =>
    `Pass ${p.pass} (${p.n_tokens > 1 ? 'prefill' : 'decode'}, ${p.n_ops} ops)`;

  // Lookup DAG node for shape/dtype
  const dagNode: DagNode | undefined = currentStep
    ? data.dag.nodes.find(n => n.id === currentStep.name)
    : undefined;

  const duration = currentStep ? currentStep.t_end - currentStep.t_start : 0;
  const heatRatio = maxDur > minDur ? (duration - minDur) / (maxDur - minDur) : 0;

  return (
    <div className="w-72 shrink-0 bg-white border-l flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between p-2 border-b">
        <span className="font-bold text-sm">Replay</span>
        <button
          className="text-gray-400 hover:text-gray-600 text-lg leading-none"
          onClick={handleStopClick}
          title="Close replay"
        >&times;</button>
      </div>

      {/* Pass selector */}
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

        {/* Expand mode */}
        <div className="flex gap-2 text-xs">
          <label className="flex items-center gap-1 cursor-pointer">
            <input
              type="radio"
              name="expandMode"
              checked={expandMode === 'keep'}
              onChange={() => onExpandModeChange('keep')}
              className="w-3 h-3"
            />
            Keep Current
          </label>
          <label className="flex items-center gap-1 cursor-pointer">
            <input
              type="radio"
              name="expandMode"
              checked={expandMode === 'auto'}
              onChange={() => onExpandModeChange('auto')}
              className="w-3 h-3"
            />
            Auto Expand
          </label>
        </div>
      </div>

      {/* Transport controls */}
      <div className="p-2 border-b space-y-2">
        <div className="flex items-center gap-1">
          {playing ? (
            <button
              className="px-3 py-1 text-xs rounded bg-gray-800 text-white hover:bg-gray-700"
              onClick={handlePause}
            >Pause</button>
          ) : (
            <button
              className="px-3 py-1 text-xs rounded bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-30"
              onClick={handlePlay}
              disabled={steps.length === 0}
            >Play</button>
          )}
          <button
            className="px-3 py-1 text-xs rounded bg-gray-200 hover:bg-gray-300"
            onClick={handleStopClick}
          >Stop</button>
          <span className="text-xs text-gray-400 ml-auto">
            {currentIndex >= 0 ? currentIndex + 1 : 0}/{steps.length}
          </span>
        </div>

        {/* Speed presets */}
        <div className="flex gap-1">
          {SPEED_PRESETS.map(s => (
            <button
              key={s}
              className={`flex-1 py-0.5 text-xs rounded ${speed === s ? 'bg-gray-800 text-white' : 'bg-gray-200'}`}
              onClick={() => handleSpeedChange(s)}
            >{s}x</button>
          ))}
        </div>

        {/* Progress bar */}
        <input
          type="range"
          min={0}
          max={Math.max(0, steps.length - 1)}
          value={Math.max(0, currentIndex)}
          onChange={e => handleSeek(Number(e.target.value))}
          className="w-full h-1 cursor-pointer"
        />
      </div>

      {/* Current op card */}
      {currentStep && (
        <div
          className="m-2 p-2 border rounded text-xs space-y-1"
          style={{ borderLeftWidth: 4, borderLeftColor: heatmapColor(heatRatio) }}
        >
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

      {/* Log queue */}
      <div ref={logRef} className="overflow-y-auto flex-1 text-xs">
        {log.map((op, i) => {
          const isActive = i === 0 && currentStep?.seq === op.seq;
          return (
            <div
              key={`${op.seq}-${i}`}
              className={`px-2 py-1 border-b cursor-pointer hover:bg-blue-50 flex gap-2 ${isActive ? 'bg-blue-100' : ''}`}
              onClick={() => handleSeek(steps.findIndex(s => s.seq === op.seq))}
            >
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
```

- [ ] **Step 2: Verify it compiles**

Run: `cd web && npx tsc --noEmit`
Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add web/src/components/ReplayPanel.tsx
git commit -m "tools: add ReplayPanel component with controls and log queue"
```

---

### Task 3: DagView Replay Integration — useEffect + Cytoscape Styles

**Files:**
- Modify: `web/src/components/DagView.tsx`

Add three new Cytoscape style selectors (`.replay-visited`, `.replay-current`,
`.replay-edge-active`) and a `useEffect` that reacts to `replayState` prop changes
by adding/removing classes on nodes and edges. No Cytoscape rebuild.

- [ ] **Step 1: Add ReplayState import and prop to DagView**

In `DagView.tsx`, add the import and update the Props interface:

```ts
// Add import at top
import type { ReplayState, ExpandMode } from './ReplayPanel';

// Update Props interface
interface Props {
  data: SummaryData;
  highlightId: string | null;
  onSelectNode: (id: string) => void;
  replayState?: ReplayState | null;
  replayExpandMode?: ExpandMode;
  onReplayActivate?: () => void;
}
```

Update the function signature:

```ts
export function DagView({ data, highlightId, onSelectNode, replayState, replayExpandMode, onReplayActivate }: Props) {
```

- [ ] **Step 2: Add three replay Cytoscape styles**

Add these three style entries to the `style` array in the `cytoscape()` constructor,
after the existing `.highlighted` selector:

```ts
{
  selector: '.replay-visited',
  style: {
    'background-opacity': 0.4,
    'border-opacity': 0.4,
  },
},
{
  selector: '.replay-current',
  style: {
    'border-color': '#dc2626',
    'border-width': 6,
    'overlay-color': '#fbbf24',
    'overlay-opacity': 0.3,
    'z-index': 999,
  },
},
{
  selector: 'edge.replay-edge-active',
  style: {
    'line-color': '#fbbf24',
    'target-arrow-color': '#fbbf24',
    'width': 4,
  },
},
```

- [ ] **Step 3: Add replay useEffect**

Add a new useEffect after the existing highlight useEffect. This handles:
1. Removing previous `.replay-current`, adding `.replay-visited`
2. Adding `.replay-current` to the current node
3. Briefly flashing incoming edges
4. Smart pan to the current node
5. Auto expand/collapse when `replayExpandMode === 'auto'`

```ts
// Replay highlight effect
useEffect(() => {
  const cy = cyRef.current;
  if (!cy || !replayState) {
    // Clear all replay classes when replay stops
    if (cy) {
      cy.nodes().removeClass('replay-visited replay-current');
      cy.edges().removeClass('replay-edge-active');
    }
    return;
  }

  const { currentNodeId, visitedNodeIds } = replayState;

  // Auto expand mode: expand current node's layer, collapse others
  if (replayExpandMode === 'auto') {
    const dagNode = data.dag.nodes.find(n => n.id === currentNodeId);
    if (dagNode?.layer && collapsed.has(dagNode.layer)) {
      // Expand only the current layer
      setCollapsed(prev => {
        const next = new Set(allLayers);
        if (dagNode.layer) next.delete(dagNode.layer);
        return next;
      });
      return; // will re-run after collapse state change triggers rebuild
    }
  }

  // Clear previous replay state
  cy.nodes().removeClass('replay-current');
  cy.edges().removeClass('replay-edge-active');

  // Apply visited state to all previously visited nodes
  cy.nodes().forEach(node => {
    const id = node.id();
    if (visitedNodeIds.has(id) && id !== currentNodeId) {
      node.addClass('replay-visited');
    }
  });

  // Highlight current node
  const currentNode = cy.getElementById(currentNodeId);
  if (currentNode.length) {
    currentNode.removeClass('replay-visited');
    currentNode.addClass('replay-current');

    // Flash incoming edges
    const incomingEdges = cy.edges().filter(e => e.data('target') === currentNodeId);
    incomingEdges.addClass('replay-edge-active');
    setTimeout(() => {
      incomingEdges.removeClass('replay-edge-active');
    }, 200);

    // Smart pan: only move if node is outside viewport
    const ext = cy.extent();
    const pos = currentNode.position();
    if (pos.x < ext.x1 || pos.x > ext.x2 || pos.y < ext.y1 || pos.y > ext.y2) {
      cy.animate({ center: { eles: currentNode }, duration: 150 });
    }
  } else if (replayExpandMode === 'keep') {
    // Node not visible (inside collapsed layer) — flash the layer node instead
    const dagNode = data.dag.nodes.find(n => n.id === currentNodeId);
    if (dagNode?.layer) {
      const layerNode = cy.getElementById(`layer:${dagNode.layer}`);
      if (layerNode.length) {
        layerNode.addClass('replay-current');
        setTimeout(() => layerNode.removeClass('replay-current'), 200);
      }
    }
  }
}, [replayState, replayExpandMode, collapsed, colorMode, layoutName, data.dag.nodes]);
```

The dependency array includes `collapsed`, `colorMode`, and `layoutName` because these
trigger Cytoscape rebuilds (destroy + recreate `cy`). Without them, the replay useEffect
wouldn't re-fire after a rebuild, leaving the new Cytoscape instance without replay classes.

- [ ] **Step 5: Add Replay button to the toolbar**

In the DagView toolbar, after the search input, add:

```tsx
{!replayState && onReplayActivate && (
  <>
    <div className="w-px h-5 bg-gray-300 mx-0.5" />
    <Btn onClick={onReplayActivate} title="Replay execution">
      Replay
    </Btn>
  </>
)}
```

Uses a dedicated `onReplayActivate` callback prop instead of overloading `onSelectNode`.

- [ ] **Step 6: Verify it compiles**

Run: `cd web && npx tsc --noEmit`
Expected: no errors (DagView callers may warn until App is updated in Task 4)

- [ ] **Step 7: Commit**

```bash
git add web/src/components/DagView.tsx
git commit -m "tools: add replay highlight integration to DagView"
```

---

### Task 4: App.tsx Integration — Conditional Rendering

**Files:**
- Modify: `web/src/App.tsx`

Add `replayActive` state. When active, render ReplayPanel instead of HotspotPanel.
Pass `replayState` down to DagView. Handle the `'__replay__'` signal from DagView
to activate replay mode.

- [ ] **Step 1: Add replay state and imports to App.tsx**

```ts
// Add imports
import { ReplayPanel, type ReplayState, type ExpandMode } from './components/ReplayPanel';

// Inside App component, add state:
const [replayActive, setReplayActive] = useState(false);
const [replayState, setReplayState] = useState<ReplayState | null>(null);
const [replayExpandMode, setReplayExpandMode] = useState<ExpandMode>('keep');
```

- [ ] **Step 2: Add replay activation handler**

No need to intercept `setHighlightId` — replay uses a dedicated callback:

```ts
const handleReplayActivate = () => {
  setReplayActive(true);
  setHighlightId(null);
};
```

- [ ] **Step 3: Update DagView rendering to pass replay props**

```tsx
{view === 'dag' && summaryData && (
  <DagView
    data={summaryData}
    highlightId={highlightId}
    onSelectNode={setHighlightId}
    replayState={replayState}
    replayExpandMode={replayExpandMode}
    onReplayActivate={handleReplayActivate}
  />
)}
```

- [ ] **Step 4: Conditional sidebar rendering**

Replace the HotspotPanel rendering with conditional logic:

```tsx
{summaryData && view !== 'compare' && (
  replayActive && view === 'dag' ? (
    <ReplayPanel
      data={summaryData}
      onReplayState={setReplayState}
      onStop={() => { setReplayActive(false); setReplayState(null); }}
      expandMode={replayExpandMode}
      onExpandModeChange={setReplayExpandMode}
    />
  ) : (
    <HotspotPanel data={summaryData} selectedId={highlightId} onSelect={setHighlightId} />
  )
)}
```

- [ ] **Step 5: Deactivate replay on view change**

Add an effect to deactivate replay when switching away from DAG view:

```ts
useEffect(() => {
  if (view !== 'dag') {
    setReplayActive(false);
    setReplayState(null);
  }
}, [view]);
```

- [ ] **Step 6: Verify it compiles and runs**

Run: `cd web && npx tsc --noEmit`
Run: `cd web && npm run dev`
Expected: app starts, DAG view shows "Replay" button in toolbar

- [ ] **Step 7: Commit**

```bash
git add web/src/App.tsx
git commit -m "tools: integrate replay mode in App with conditional sidebar"
```

---

### Task 5: Manual Integration Testing

**Files:** None (testing only)

- [ ] **Step 1: Start dev server with test data**

```bash
cd tools/trace-analyzer
python -m trace_analyzer serve --data-dir <path-to-trace-data>
```

Open browser to the dev server URL.

- [ ] **Step 2: Test basic replay flow**

1. Select a summary file
2. Switch to DAG view
3. Click "Replay" button in toolbar → ReplayPanel appears, HotspotPanel hidden
4. Select a pass from dropdown
5. Click Play → ops highlight one by one on DAG
6. Verify visited nodes dim, current node has red border + yellow overlay
7. Verify incoming edges flash yellow briefly
8. Verify smart pan when node is off-screen

- [ ] **Step 3: Test transport controls**

1. Pause → playback stops, button changes to Play
2. Resume (Play while paused) → continues from where it stopped
3. Speed presets → faster/slower playback
4. Drag progress bar → seek to position
5. Click log entry → seek to that step
6. Stop → clears all highlights, returns to HotspotPanel

- [ ] **Step 4: Test expand modes**

1. "Keep Current" mode with all layers collapsed → layer node flashes when current op is inside
2. "Auto Expand" mode → current op's layer auto-expands, others collapse
3. Verify no crash when switching modes during playback

- [ ] **Step 5: Test edge cases**

1. Empty pass (0 ops) → Play disabled, message shown
2. Change pass during playback → auto-stops and resets
3. Switch view tab during replay → replay deactivated
4. Close replay via X button → returns to HotspotPanel

- [ ] **Step 6: Final commit and tag**

```bash
git add -A
git commit -m "tools: complete trace replay feature"
git tag v0.3.0-trace-replay
```

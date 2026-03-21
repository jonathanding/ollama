# Trace Replay: Computation Graph Playback

## Overview

Animate the execution of a single inference pass on the DAG view, stepping through
ops in execution order to visualize data flow through the computation graph.
Built entirely in the frontend using existing `summary.json` data — no backend changes.

## Data Source

`summary.json` already contains everything needed:

- `timeline_ops[]`: each op has `pass`, `seq` (execution order), `name` (= DAG node id),
  `op`, `backend`, `t_start`, `t_end`
- `dag.nodes[]` / `dag.edges[]`: graph structure with node ids matching `timeline_ops.name`

Replay steps for a given pass — reuses `TimelineOp` directly since `name` is
already the DAG node id:
```ts
type ReplayStep = TimelineOp;
// duration computed on the fly: step.t_end - step.t_start

const steps: ReplayStep[] = data.timeline_ops
  .filter(op => op.pass === selectedPass)
  .sort((a, b) => a.seq - b.seq);
```

**Pass type heuristic**: `n_tokens > 1` → prefill, `n_tokens === 1` → decode.
Available from `data.timing.per_pass[].n_tokens`.

## Architecture

### ReplayController (pure TS class)

No React dependency. Manages playback state and step progression.

```ts
interface ReplayCallbacks {
  onStep: (step: TimelineOp, index: number) => void;
  onFinish: () => void;
}

class ReplayController {
  state: 'idle' | 'playing' | 'paused';
  steps: TimelineOp[];
  currentIndex: number;
  speed: number;          // 1, 2, 5, 10, 20

  play(): void;
  pause(): void;
  resume(): void;
  stop(): void;
  setSpeed(n: number): void;
  seekTo(index: number): void;  // updates highlight immediately
}
```

Timing: `setInterval(BASE_INTERVAL / speed)` where `BASE_INTERVAL = 80ms`.
At 20x this is 4ms; browser will naturally clamp to ~4ms minimum timer resolution.
No requestAnimationFrame needed — setInterval is sufficient for step progression.

Speed presets: 1x, 2x, 5x, 10x, 20x.

### Time Expression (Hybrid Mode)

Steps advance at equal intervals (smooth playback). Op duration is expressed
visually:

- **Color intensity**: expensive ops get a warmer heatmap color on the highlight
  overlay. Ratio = `(duration - minDuration) / (maxDuration - minDuration)` across
  all steps in the current pass. Uses existing `heatmapColor()` from `colorScale.ts`.
- **Fade duration**: expensive ops keep their highlight longer via CSS transition;
  cheap ops fade quickly
- The current-op detail card always shows the real nanosecond duration

### DAG Highlight — Three Node States

| State     | Visual                                   |
|-----------|------------------------------------------|
| Unvisited | Normal color (no modification)           |
| Visited   | Semi-transparent overlay, reduced saturation |
| Current   | Red border + yellow overlay (reuses existing highlight style) |

Cytoscape style definitions for replay classes:

```ts
{ selector: '.replay-visited',
  style: { 'background-opacity': 0.4, 'border-opacity': 0.4 } },

{ selector: '.replay-current',
  style: {  // same as existing .highlighted
    'border-color': '#dc2626', 'border-width': 6,
    'overlay-color': '#fbbf24', 'overlay-opacity': 0.3, 'z-index': 999 } },

{ selector: 'edge.replay-edge-active',
  style: { 'line-color': '#fbbf24', 'target-arrow-color': '#fbbf24', 'width': 4 } },
```

Each step:
1. Previous node: `.replay-current` → `.replay-visited`
2. Current node: add `.replay-current`
3. Current node's incoming edges: brief flash (`.replay-edge-active`, removed after
   200ms). Incoming edges = `dag.edges.filter(e => e.to === currentNodeId)`.
   Cytoscape edge id = `${edge.from}-${edge.to}`.

All via `cy.getElementById().addClass/removeClass` — no Cytoscape rebuild.

### Two Expand/Collapse Modes

**Mode 1 — Keep Current**: No automatic expand/collapse. If the current op is
inside a collapsed layer, the layer summary node flashes instead. Op details
still show in the log queue.

**Mode 2 — Auto Expand**: When the current op's layer differs from the
previously expanded one:
1. Collapse all expanded layers
2. Expand only the current op's layer
3. Single `setCollapsed` update → one Cytoscape rebuild
4. Re-apply `.replay-visited` classes to all previously visited nodes after rebuild
5. Smart pan to the current node

Consecutive ops in the same layer don't trigger rebuild. Animation duration
kept short (150ms) for smooth transitions.

### Smart Pan

Check if node position is within the visible viewport rectangle (`cy.extent()`
returns `{ x1, y1, x2, y2 }` in model coordinates). If the node's position
falls outside these bounds, `cy.animate({ center: node }, { duration: 150 })`.
If inside, no movement. Reuses the same logic already in DagView's highlight
effect.

## ReplayPanel (Right Sidebar)

Replaces HotspotPanel when replay is active. Same outer dimensions (`w-72
shrink-0`), but internal layout is three stacked sections instead of a
scrollable list.

### Top: Controls
- **Pass selector**: dropdown listing all passes with type label
  (prefill/decode, derived from `per_pass[].n_tokens > 1`) and op count
- **Expand mode**: radio toggle — "Keep Current" / "Auto Expand"
- **Transport**: Play ▶ / Pause ⏸ / Stop ⏹ (play/pause share one button)
- **Speed**: preset buttons — 1x, 2x, 5x, 10x, 20x (active one highlighted)
- **Progress bar**: draggable, shows `currentIndex / totalSteps`, click to seek

### Middle: Current Op Card
Fixed-height card showing the current step:
- Op type (large text)
- Tensor name (mono font)
- Backend + duration
- Shape + dtype (looked up from `dag.nodes` by matching id)
- Left border color = heatmap by duration

### Bottom: Op Log Queue
- Newest entry at top, auto-scrolls
- Shows most recent ~50 ops; earlier ones collapsed as "... N earlier ops"
- Each row: `#seq  OP_TYPE  tensor_name  duration  backend`
- Current step row has highlight background
- Click any row → `seekTo` that step

## App Integration

### State

`App.tsx` adds:
```ts
const [replayActive, setReplayActive] = useState(false);
```

### Conditional Rendering

```tsx
{replayActive
  ? <ReplayPanel data={summaryData} onStop={() => setReplayActive(false)} ... />
  : <HotspotPanel ... />
}
```

### Entry Point

DAG toolbar gets a `▶ Replay` button. Clicking it sets `replayActive = true`.

### Exit

- Stop button in ReplayPanel
- X close button in ReplayPanel header
- On exit: clear all `.replay-*` classes from Cytoscape, restore HotspotPanel

### Data Flow

```
App
├─ DagView
│   props: replayState?: ReplayState
│   useEffect: operates on cyRef to add/remove classes (no rebuild)
│
├─ HotspotPanel (if !replayActive)
│
└─ ReplayPanel (if replayActive)
    ├─ owns ReplayController instance
    └─ onStep → updates App state → flows to DagView props
```

```ts
interface ReplayState {
  currentNodeId: string;
  visitedNodeIds: Set<string>;
}
```

Edge activation is computed inside DagView's replay useEffect from
`dag.edges.filter(e => e.to === currentNodeId)` — no need to pass edge ids
through props.

## File Structure

```
web/src/
├─ utils/
│   └─ ReplayController.ts     # Pure TS, no React
├─ components/
│   ├─ ReplayPanel.tsx          # Controls + detail card + log queue
│   └─ DagView.tsx              # New replay useEffect + 3 new CSS classes
└─ App.tsx                      # replayActive state + conditional render
```

No Python backend changes. No changes to summary.json schema.

## Edge Cases

- **Empty pass**: pass with 0 ops → disable Play button, show message
- **Node not in DAG**: timeline_ops may reference ops not in the DAG
  (different pass). Filter steps to only include ops with matching DAG node ids.
- **Rapid speed**: at 20x with 80ms base, interval is 4ms. Browser clamps
  setInterval to ~4ms minimum; this is acceptable (250 steps/sec max).
- **Seek while paused**: update highlight immediately without resuming playback
- **Pass change during playback**: auto-stop and reset to idle before loading
  new pass data
- **Layout/color change during replay**: preserve replay state (visited set +
  current index). After Cytoscape rebuild, re-apply `.replay-visited` to all
  visited nodes and `.replay-current` to the current node.
- **Window resize during playback**: Cytoscape auto-handles; no special logic needed

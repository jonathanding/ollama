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

Replay steps for a given pass:
```
timeline_ops.filter(op.pass === selectedPass).sort(by seq)
→ ReplayStep[] = [{ nodeId, op, backend, t_start, t_end, duration_ns }]
```

## Architecture

### ReplayController (pure TS class)

No React dependency. Manages playback state and step progression.

```
State:    idle | playing | paused
Fields:   steps[], currentIndex, speed, intervalId
Methods:  play(), pause(), resume(), stop(), setSpeed(n), seekTo(index)
Callback: onStep(step, index) — fires each time the current step advances
```

Interval: fixed `BASE_INTERVAL / speed` (e.g. 80ms at 1x, 4ms at 20x).
Speed presets: 1x, 2x, 5x, 10x, 20x.

### Time Expression (Hybrid Mode)

Steps advance at equal intervals (smooth playback). Op duration is expressed
visually:

- **Color intensity**: expensive ops get a warmer heatmap color on highlight
- **Fade duration**: expensive ops keep their highlight longer via CSS transition;
  cheap ops fade quickly
- The current-op detail card always shows the real nanosecond duration

### DAG Highlight — Three Node States

| State     | Visual                                   |
|-----------|------------------------------------------|
| Unvisited | Normal color (no modification)           |
| Visited   | Semi-transparent overlay, reduced saturation (`.replay-visited` class) |
| Current   | Red border + yellow overlay (`.replay-current` class, reuses existing highlight style) |

Each step:
1. Previous node: `.replay-current` → `.replay-visited`
2. Current node: add `.replay-current`
3. Current node's incoming edges: brief flash (`.replay-edge-active`, removed after 200ms)

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
4. Smart pan to the current node

Consecutive ops in the same layer don't trigger rebuild. Animation duration
kept short (150ms) for smooth transitions.

### Smart Pan

Reuses existing logic: check if node is within `cy.extent()`. If outside
viewport, `cy.animate({ center: node }, { duration: 150 })`. If inside, no
movement.

## ReplayPanel (Right Sidebar)

Replaces HotspotPanel when replay is active. Same dimensions (`w-72 shrink-0`).
Three vertical sections:

### Top: Controls
- **Pass selector**: dropdown listing all passes with type label
  (prefill/decode) and op count
- **Expand mode**: radio toggle — "Keep Current" / "Auto Expand"
- **Transport**: Play ▶ / Pause ⏸ / Stop ⏹ (play/pause share one button)
- **Speed**: preset buttons — 1x, 2x, 5x, 10x, 20x (active one highlighted)
- **Progress bar**: draggable, shows `currentIndex / totalSteps`, click to seek

### Middle: Current Op Card
Fixed-height card showing the current step:
- Op type (large text)
- Tensor name (mono font)
- Backend + duration
- Shape + dtype
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
│   props: replayState?: { currentNodeId, visitedNodeIds, activeEdgeIds }
│   useEffect: operates on cyRef to add/remove classes (no rebuild)
│
├─ HotspotPanel (if !replayActive)
│
└─ ReplayPanel (if replayActive)
    ├─ owns ReplayController instance
    └─ onStep → updates App state → flows to DagView props
```

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
- **Rapid speed**: at 20x with 80ms base, interval is 4ms. `requestAnimationFrame`
  naturally caps at ~16ms. Use rAF instead of setInterval at high speeds.
- **Seek while paused**: update highlight immediately without resuming playback
- **Window resize during playback**: Cytoscape auto-handles; no special logic needed

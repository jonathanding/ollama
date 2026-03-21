import cytoscape from 'cytoscape';
import type { ElementDefinition } from 'cytoscape';

/**
 * Two-pass compound-aware tree layout:
 *
 * 1. Build a "meta graph" where every layer is a single node. Run breadthfirst
 *    to determine tree topology and relative positions.
 * 2. Scale positions so that expanded layers (which become large rectangles)
 *    don't overlap each other.
 * 3. Arrange children inside expanded layers in a grid.
 *
 * Returns { nodeId → {x,y} } for Cytoscape's `preset` layout.
 */
export function computeTreeCompoundPositions(
  elements: ElementDefinition[],
): Map<string, { x: number; y: number }> {
  const positions = new Map<string, { x: number; y: number }>();

  // --- Classify nodes ---
  const parentIds = new Set<string>();
  const childrenOf = new Map<string, ElementDefinition[]>();

  for (const el of elements) {
    if (el.data.source) continue;
    const parent = el.data.parent as string | undefined;
    if (parent) {
      parentIds.add(parent);
      if (!childrenOf.has(parent)) childrenOf.set(parent, []);
      childrenOf.get(parent)!.push(el);
    }
  }

  // --- Build meta graph ---
  const CHILD_CELL_W = 55;
  const CHILD_CELL_H = 50;
  const PADDING = 30; // padding inside compound box

  // Compute the footprint each meta-node needs
  const metaNodeSize = new Map<string, { w: number; h: number }>();
  const metaNodes: ElementDefinition[] = [];

  for (const el of elements) {
    if (el.data.source) continue;
    if (el.data.parent) continue; // skip children
    const id = el.data.id as string;

    if (parentIds.has(id)) {
      const nChildren = childrenOf.get(id)?.length ?? 1;
      const cols = Math.ceil(Math.sqrt(nChildren));
      const rows = Math.ceil(nChildren / cols);
      const w = cols * CHILD_CELL_W + PADDING * 2;
      const h = rows * CHILD_CELL_H + PADDING * 2;
      metaNodeSize.set(id, { w, h });
    } else {
      metaNodeSize.set(id, { w: 70, h: 55 });
    }
    metaNodes.push({ data: { id } });
  }

  // Remap child edges to parent
  const childToParent = new Map<string, string>();
  for (const el of elements) {
    if (el.data.source) continue;
    if (el.data.parent) childToParent.set(el.data.id as string, el.data.parent as string);
  }

  const metaEdgeSet = new Set<string>();
  const metaEdges: ElementDefinition[] = [];
  for (const el of elements) {
    if (!el.data.source) continue;
    const src = childToParent.get(el.data.source) ?? el.data.source;
    const tgt = childToParent.get(el.data.target) ?? el.data.target;
    if (src === tgt) continue;
    const key = `${src}->${tgt}`;
    if (metaEdgeSet.has(key)) continue;
    metaEdgeSet.add(key);
    metaEdges.push({ data: { source: src, target: tgt } });
  }

  // --- Pass 1: breadthfirst on meta graph ---
  const metaCy = cytoscape({
    headless: true,
    elements: [...metaNodes, ...metaEdges],
    layout: { name: 'breadthfirst', directed: true, spacingFactor: 1.5 } as any,
  });

  // Extract raw positions and determine depth levels
  const rawPos = new Map<string, { x: number; y: number }>();
  metaCy.nodes().forEach(n => {
    rawPos.set(n.id(), { ...n.position() });
  });
  metaCy.destroy();

  if (rawPos.size === 0) return positions;

  // --- Scale positions to avoid overlap ---
  // Group nodes by their Y coordinate (depth level in breadthfirst)
  // Use rounded Y to handle floating point
  const yPrecision = 0.1;
  const depthGroups = new Map<number, string[]>();
  for (const [id, pos] of rawPos) {
    const roundedY = Math.round(pos.y / yPrecision) * yPrecision;
    if (!depthGroups.has(roundedY)) depthGroups.set(roundedY, []);
    depthGroups.get(roundedY)!.push(id);
  }

  // For each depth level, compute the total width needed and spacing
  // Vertical spacing: max height of any node at adjacent levels + gap
  const sortedDepths = [...depthGroups.keys()].sort((a, b) => a - b);

  // Compute max height at each depth
  const depthMaxH = new Map<number, number>();
  for (const [depth, ids] of depthGroups) {
    let maxH = 0;
    for (const id of ids) {
      maxH = Math.max(maxH, metaNodeSize.get(id)?.h ?? 55);
    }
    depthMaxH.set(depth, maxH);
  }

  // Vertical: assign absolute Y based on cumulative heights + gaps
  const GAP_V = 40;
  const depthY = new Map<number, number>();
  let cumY = 0;
  for (let i = 0; i < sortedDepths.length; i++) {
    const depth = sortedDepths[i];
    depthY.set(depth, cumY);
    cumY += (depthMaxH.get(depth) ?? 55) + GAP_V;
  }

  // Horizontal: at each depth, space nodes by their widths + gap
  const GAP_H = 50;
  for (const [depth, ids] of depthGroups) {
    // Sort by original X to preserve breadthfirst ordering
    ids.sort((a, b) => (rawPos.get(a)!.x - rawPos.get(b)!.x));

    // Compute total width
    let totalW = 0;
    for (const id of ids) {
      totalW += metaNodeSize.get(id)?.w ?? 70;
    }
    totalW += (ids.length - 1) * GAP_H;

    // Assign X centered around 0
    let cx = -totalW / 2;
    for (const id of ids) {
      const w = metaNodeSize.get(id)?.w ?? 70;
      const nodeX = cx + w / 2;
      const nodeY = depthY.get(depth) ?? 0;
      positions.set(id, { x: nodeX, y: nodeY });
      cx += w + GAP_H;
    }
  }

  // --- Pass 2: position children inside expanded layers ---
  for (const [parentId, children] of childrenOf) {
    const center = positions.get(parentId);
    if (!center) continue;

    const n = children.length;
    const cols = Math.ceil(Math.sqrt(n));
    const totalW = cols * CHILD_CELL_W;
    const totalRows = Math.ceil(n / cols);
    const totalH = totalRows * CHILD_CELL_H;

    children.forEach((child, i) => {
      const col = i % cols;
      const row = Math.floor(i / cols);
      positions.set(child.data.id as string, {
        x: center.x - totalW / 2 + col * CHILD_CELL_W + CHILD_CELL_W / 2,
        y: center.y - totalH / 2 + row * CHILD_CELL_H + CHILD_CELL_H / 2,
      });
    });
  }

  return positions;
}

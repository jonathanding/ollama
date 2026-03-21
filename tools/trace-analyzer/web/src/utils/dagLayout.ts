import type { DagData } from '../types/trace';
import type { ElementDefinition } from 'cytoscape';
import { backendColor, heatmapColor, opColor } from './colorScale';
import { formatNs } from './dataSize';

export type ColorMode = 'backend' | 'heatmap' | 'op';

export function buildCytoscapeElements(
  dag: DagData,
  colorMode: ColorMode,
  collapsed: Set<string>,
): ElementDefinition[] {
  const elements: ElementDefinition[] = [];

  // Use reduce to avoid stack overflow on large arrays
  const maxNs = dag.nodes.reduce((m, n) => Math.max(m, n.ns), 1);

  const layers = new Set<string>();
  const layerTotals = new Map<string, number>();
  for (const node of dag.nodes) {
    if (node.layer) {
      layers.add(node.layer);
      layerTotals.set(node.layer, (layerTotals.get(node.layer) ?? 0) + node.ns);
    }
  }
  let maxLayerTotal = 1;
  for (const v of layerTotals.values()) {
    if (v > maxLayerTotal) maxLayerTotal = v;
  }

  // Layer compound nodes
  for (const layer of layers) {
    const totalNs = layerTotals.get(layer) ?? 0;
    const bgColor = colorMode === 'heatmap'
      ? heatmapColor(totalNs / maxLayerTotal)
      : '#e5e7eb';

    elements.push({
      data: {
        id: `layer:${layer}`,
        label: `${layer}\n${formatNs(totalNs)}`,
        isLayer: true,
        bgColor,
      },
    });
  }

  // Op nodes (skip if their layer is collapsed)
  for (const node of dag.nodes) {
    const layerId = node.layer ? `layer:${node.layer}` : undefined;
    if (layerId && collapsed.has(node.layer!)) continue;

    const bgColor = colorMode === 'heatmap'
      ? heatmapColor(node.ns / maxNs)
      : colorMode === 'op'
      ? opColor(node.op)
      : backendColor(node.backend);

    const size = Math.max(30, Math.log2(node.ns + 1) * 5);
    // Truncate long names for display
    const shortId = node.id.length > 20 ? node.id.slice(0, 18) + '..' : node.id;

    elements.push({
      data: {
        ...node,
        id: node.id,
        label: `${node.op}\n${shortId}`,
        parent: layerId,
        bgColor,
        isCopy: node.is_copy ? 'yes' : 'no',
        borderWidth: node.is_copy ? 3 : 1,
        nodeWidth: size,
        nodeHeight: size,
      },
    });
  }

  // Edges — promote to layer-level when endpoints are collapsed
  const nodeMap = new Map(dag.nodes.map(n => [n.id, n]));
  const addedEdges = new Set<string>();

  for (const edge of dag.edges) {
    const fromNode = nodeMap.get(edge.from);
    const toNode = nodeMap.get(edge.to);
    if (!fromNode || !toNode) continue;

    const fromCollapsed = fromNode.layer && collapsed.has(fromNode.layer);
    const toCollapsed = toNode.layer && collapsed.has(toNode.layer);

    const source = fromCollapsed ? `layer:${fromNode.layer}` : edge.from;
    const target = toCollapsed ? `layer:${toNode.layer}` : edge.to;

    if (source === target) continue;

    const edgeKey = `${source}->${target}`;
    if (addedEdges.has(edgeKey)) continue;
    addedEdges.add(edgeKey);

    elements.push({
      data: {
        source,
        target,
        est_bytes: edge.est_bytes,
        edgeWidth: Math.max(1, Math.log2(edge.est_bytes + 1) * 0.5),
      },
    });
  }

  return elements;
}

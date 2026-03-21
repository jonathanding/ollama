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

  const maxNs = Math.max(...dag.nodes.map(n => n.ns), 1);

  const layers = new Set<string>();
  const layerTotals = new Map<string, number>();
  for (const node of dag.nodes) {
    if (node.layer) {
      layers.add(node.layer);
      layerTotals.set(node.layer, (layerTotals.get(node.layer) ?? 0) + node.ns);
    }
  }
  const maxLayerTotal = Math.max(...layerTotals.values(), 1);

  // Layer compound nodes
  for (const layer of layers) {
    const totalNs = layerTotals.get(layer) ?? 0;
    const bgColor = colorMode === 'heatmap'
      ? heatmapColor(totalNs / maxLayerTotal)
      : '#e5e7eb';

    elements.push({
      data: {
        id: `layer:${layer}`,
        label: `${layer} (${formatNs(totalNs)})`,
        isLayer: true,
        bgColor,
      },
    });
  }

  // Nodes
  const nodeIdSet = new Set<string>();
  for (const node of dag.nodes) {
    const layerId = node.layer ? `layer:${node.layer}` : undefined;
    if (layerId && collapsed.has(node.layer!)) continue;

    const bgColor = colorMode === 'heatmap'
      ? heatmapColor(node.ns / maxNs)
      : colorMode === 'op'
      ? opColor(node.op)
      : backendColor(node.backend);

    const size = Math.max(30, Math.log2(node.ns + 1) * 5);

    nodeIdSet.add(node.id);
    elements.push({
      data: {
        ...node,
        id: node.id,
        label: `${node.op}\n${node.id}`,
        parent: layerId,
        bgColor,
        borderStyle: node.is_copy ? 'dashed' : 'solid',
        borderColor: node.is_copy ? '#ef4444' : '#6b7280',
        borderWidth: node.is_copy ? 3 : 1,
        nodeWidth: size,
        nodeHeight: size,
      },
    });
  }

  // Edges — when both endpoints are visible, connect directly.
  // When one or both endpoints are in a collapsed layer, promote edge to layer node.
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

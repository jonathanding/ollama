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
        collapsed: collapsed.has(layer),
      },
      style: { 'background-color': bgColor },
    });
  }

  for (const node of dag.nodes) {
    const layerId = node.layer ? `layer:${node.layer}` : undefined;
    if (layerId && collapsed.has(node.layer!)) continue;

    const bgColor = colorMode === 'heatmap'
      ? heatmapColor(node.ns / maxNs)
      : colorMode === 'op'
      ? opColor(node.op)
      : backendColor(node.backend);

    elements.push({
      data: {
        ...node,
        id: node.id,
        label: `${node.op}\n${node.id}`,
        parent: layerId,
      },
      style: {
        'background-color': bgColor,
        'border-style': node.is_copy ? 'dashed' : 'solid',
        'border-color': node.is_copy ? '#ef4444' : '#6b7280',
        'border-width': node.is_copy ? 3 : 1,
        width: Math.max(30, Math.log2(node.ns + 1) * 5),
        height: Math.max(30, Math.log2(node.ns + 1) * 5),
      },
    });
  }

  const nodeMap = new Map(dag.nodes.map(n => [n.id, n]));

  for (const edge of dag.edges) {
    const fromNode = nodeMap.get(edge.from);
    const toNode = nodeMap.get(edge.to);
    if (fromNode?.layer && collapsed.has(fromNode.layer)) continue;
    if (toNode?.layer && collapsed.has(toNode.layer)) continue;

    elements.push({
      data: {
        source: edge.from,
        target: edge.to,
        est_bytes: edge.est_bytes,
      },
      style: {
        width: Math.max(1, Math.log2(edge.est_bytes + 1) * 0.5),
      },
    });
  }

  return elements;
}

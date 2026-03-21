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

  const maxNs = dag.nodes.reduce((m, n) => Math.max(m, n.ns), 1);

  // Compute layer stats
  const layerTotals = new Map<string, number>();
  const layerOpCounts = new Map<string, number>();
  const layerTopOp = new Map<string, string>();
  const layerTopOpNs = new Map<string, number>();
  for (const node of dag.nodes) {
    if (!node.layer) continue;
    layerTotals.set(node.layer, (layerTotals.get(node.layer) ?? 0) + node.ns);
    layerOpCounts.set(node.layer, (layerOpCounts.get(node.layer) ?? 0) + 1);
    const prev = layerTopOpNs.get(node.layer) ?? 0;
    if (node.ns > prev) {
      layerTopOp.set(node.layer, node.op);
      layerTopOpNs.set(node.layer, node.ns);
    }
  }
  let maxLayerTotal = 1;
  for (const v of layerTotals.values()) {
    if (v > maxLayerTotal) maxLayerTotal = v;
  }

  const allLayers = new Set<string>();
  for (const node of dag.nodes) {
    if (node.layer) allLayers.add(node.layer);
  }

  // Track visible node IDs for edge filtering
  const visibleNodeIds = new Set<string>();

  for (const layer of allLayers) {
    const totalNs = layerTotals.get(layer) ?? 0;
    const nOps = layerOpCounts.get(layer) ?? 0;
    const topOp = layerTopOp.get(layer) ?? '';
    const layerId = `layer:${layer}`;

    if (collapsed.has(layer)) {
      // Collapsed: single summary node (no compound, no children)
      const bgColor = colorMode === 'heatmap'
        ? heatmapColor(totalNs / maxLayerTotal)
        : '#bfdbfe'; // blue-200
      visibleNodeIds.add(layerId);
      elements.push({
        data: {
          id: layerId,
          label: `${layer}\n${nOps} ops · ${formatNs(totalNs)}\ntop: ${topOp}`,
          isLayer: true,
          bgColor,
          nodeWidth: 70,
          nodeHeight: 55,
          borderWidth: 2,
        },
      });
    } else {
      // Expanded: compound parent node + child op nodes
      const bgColor = colorMode === 'heatmap'
        ? heatmapColor(totalNs / maxLayerTotal)
        : '#dbeafe'; // light blue
      visibleNodeIds.add(layerId);
      elements.push({
        data: {
          id: layerId,
          label: `${layer} (${nOps} ops · ${formatNs(totalNs)})`,
          isLayer: true,
          bgColor,
        },
      });
    }
  }

  // Op nodes
  for (const node of dag.nodes) {
    if (node.layer && collapsed.has(node.layer)) continue;

    const bgColor = colorMode === 'heatmap'
      ? heatmapColor(node.ns / maxNs)
      : colorMode === 'op'
      ? opColor(node.op)
      : backendColor(node.backend);

    const size = Math.max(25, Math.log2(node.ns + 1) * 4);
    const shortId = node.id.length > 20 ? node.id.slice(0, 18) + '..' : node.id;
    const layerId = node.layer ? `layer:${node.layer}` : undefined;

    visibleNodeIds.add(node.id);
    elements.push({
      data: {
        ...node,
        id: node.id,
        label: `${node.op}\n${shortId}`,
        parent: layerId, // compound grouping for expanded layers
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
    if (!visibleNodeIds.has(source) || !visibleNodeIds.has(target)) continue;

    const edgeKey = `${source}->${target}`;
    if (addedEdges.has(edgeKey)) continue;
    addedEdges.add(edgeKey);

    elements.push({
      data: {
        source,
        target,
        est_bytes: edge.est_bytes,
        edgeWidth: Math.max(0.5, Math.log2(edge.est_bytes + 1) * 0.15),
      },
    });
  }

  return elements;
}

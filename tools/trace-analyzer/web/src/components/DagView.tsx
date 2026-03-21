import { useRef, useEffect, useState, useCallback } from 'react';
import cytoscape, { type Core } from 'cytoscape';
import fcose from 'cytoscape-fcose';
import type { SummaryData, DagNode } from '../types/trace';
import { buildCytoscapeElements, type ColorMode } from '../utils/dagLayout';
import { computeTreeCompoundPositions } from '../utils/treeCompoundLayout';
import { ColorToggle } from './ColorToggle';
import { NodeDetail } from './NodeDetail';
import { HeatmapHistogram } from './HeatmapHistogram';
import type { ReplayState, ExpandMode } from './ReplayPanel';

cytoscape.use(fcose);

type LayoutName = 'tree' | 'fcose' | 'circle' | 'grid' | 'concentric';

const LAYOUTS: { name: LayoutName; label: string }[] = [
  { name: 'tree', label: 'Tree' },
  { name: 'fcose', label: 'Force' },
  { name: 'circle', label: 'Circle' },
  { name: 'grid', label: 'Grid' },
  { name: 'concentric', label: 'Concentric' },
];

function getLayoutConfig(name: LayoutName, elements: any[]): any {
  switch (name) {
    case 'tree': {
      // Two-pass: macro breadthfirst + micro grid inside expanded layers
      const positions = computeTreeCompoundPositions(elements);
      return {
        name: 'preset',
        positions: (node: any) => positions.get(node.id()) ?? { x: 0, y: 0 },
        animate: false,
      };
    }
    case 'fcose':
      return {
        name: 'fcose',
        animate: false,
        quality: 'default',
        nodeDimensionsIncludeLabels: true,
        idealEdgeLength: 60,
        nodeRepulsion: () => 4500,
        edgeElasticity: () => 0.45,
        gravity: 0.25,
        gravityRange: 3.8,
        numIter: 2500,
        tilingPaddingVertical: 10,
        tilingPaddingHorizontal: 10,
        randomize: false,
      };
    case 'circle':
      return { name: 'circle', animate: false };
    case 'grid':
      return { name: 'grid', animate: false, condense: true };
    case 'concentric':
      return {
        name: 'concentric',
        animate: false,
        concentric: (n: any) => n.degree(),
        levelWidth: () => 2,
      };
    default:
      return { name, animate: false };
  }
}

interface Props {
  data: SummaryData;
  highlightId: string | null;
  onSelectNode: (id: string) => void;
  replayState?: ReplayState | null;
  replayExpandMode?: ExpandMode;
  onReplayActivate?: () => void;
}

export function DagView({ data, highlightId, onSelectNode, replayState, replayExpandMode, onReplayActivate }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const [colorMode, setColorMode] = useState<ColorMode>('backend');
  const [layoutName, setLayoutName] = useState<LayoutName>('tree');
  const allLayers = new Set(data.dag.nodes.map(n => n.layer).filter(Boolean) as string[]);
  const [collapsed, setCollapsed] = useState<Set<string>>(() => new Set(allLayers));
  const [selectedNode, setSelectedNode] = useState<DagNode | null>(null);
  const [search, setSearch] = useState('');

  const viewportRef = useRef<{ zoom: number; pan: { x: number; y: number } } | null>(null);
  const fitAfterLayoutRef = useRef(false);

  const allCollapsed = collapsed.size === allLayers.size;

  useEffect(() => {
    if (!containerRef.current) return;

    if (cyRef.current) {
      viewportRef.current = {
        zoom: cyRef.current.zoom(),
        pan: { ...cyRef.current.pan() },
      };
    }

    const elements = buildCytoscapeElements(data.dag, colorMode, collapsed);
    const layoutConfig = getLayoutConfig(layoutName, elements);

    const cy = cytoscape({
      container: containerRef.current,
      elements,
      layout: layoutConfig,
      style: [
        {
          selector: 'node',
          style: {
            'label': 'data(label)',
            'font-size': 10,
            'text-wrap': 'wrap',
            'text-valign': 'center',
            'text-halign': 'center',
            'background-color': 'data(bgColor)',
          },
        },
        {
          selector: 'node[borderWidth]',
          style: {
            'width': 'data(nodeWidth)',
            'height': 'data(nodeHeight)',
            'border-width': 'data(borderWidth)',
            'border-color': '#6b7280',
            'border-style': 'solid',
          },
        },
        {
          selector: 'node[isLayer][borderWidth]',
          style: {
            'shape': 'round-rectangle',
            'font-weight': 'bold',
            'border-color': '#3b82f6',
            'border-style': 'solid',
          },
        },
        {
          selector: ':parent',
          style: {
            'text-valign': 'top',
            'text-halign': 'center',
            'font-size': 11,
            'font-weight': 'bold',
            'background-color': 'data(bgColor)',
            'background-opacity': 0.4,
            'border-width': 2,
            'border-color': '#60a5fa',
            'border-style': 'solid',
            'padding': '15px' as any,
          },
        },
        {
          selector: 'node[isCopy="yes"]',
          style: {
            'border-style': 'dashed',
            'border-color': '#ef4444',
          },
        },
        {
          selector: 'edge',
          style: {
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle',
            'arrow-scale': 0.8,
            'line-color': '#9ca3af',
            'target-arrow-color': '#9ca3af',
            'width': 'data(edgeWidth)',
          },
        },
        {
          selector: '.highlighted',
          style: {
            'border-color': '#dc2626',
            'border-width': 6,
            'overlay-color': '#fbbf24',
            'overlay-opacity': 0.3,
            'z-index': 999,
          },
        },
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
      ],
    });

    if (fitAfterLayoutRef.current) {
      cy.fit(undefined, 30);
      fitAfterLayoutRef.current = false;
      viewportRef.current = null;
    } else if (viewportRef.current) {
      cy.zoom(viewportRef.current.zoom);
      cy.pan(viewportRef.current.pan);
      viewportRef.current = null;
    }

    cy.on('tap', 'node', (evt) => {
      const nodeData = evt.target.data();
      if (nodeData.isLayer) {
        const layer = nodeData.id.replace('layer:', '');
        setCollapsed(prev => {
          const next = new Set(prev);
          if (next.has(layer)) next.delete(layer); else next.add(layer);
          return next;
        });
      } else {
        const dagNode = data.dag.nodes.find(n => n.id === nodeData.id);
        if (dagNode) { setSelectedNode(dagNode); onSelectNode(dagNode.id); }
      }
    });

    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        setSelectedNode(null);
        onSelectNode('');
      }
    });

    cyRef.current = cy;
    return () => { cy.destroy(); };
  }, [data.dag, colorMode, collapsed, layoutName]);

  useEffect(() => {
    if (!highlightId) return;
    const dagNode = data.dag.nodes.find(n => n.id === highlightId);
    if (dagNode?.layer && collapsed.has(dagNode.layer)) {
      setCollapsed(prev => {
        const next = new Set(prev);
        next.delete(dagNode.layer!);
        return next;
      });
      return;
    }
    if (dagNode) setSelectedNode(dagNode);
    const cy = cyRef.current;
    if (!cy) return;
    cy.nodes().removeClass('highlighted');
    const node = cy.getElementById(highlightId);
    if (node.length) {
      node.addClass('highlighted');
      const ext = cy.extent();
      const pos = node.position();
      if (pos.x < ext.x1 || pos.x > ext.x2 || pos.y < ext.y1 || pos.y > ext.y2) {
        cy.animate({ center: { eles: node }, duration: 200 });
      }
    }
  }, [highlightId, collapsed]);

  // Replay highlight effect
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !replayState) {
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
        setCollapsed(() => {
          const next = new Set(allLayers);
          if (dagNode.layer) next.delete(dagNode.layer);
          return next;
        });
        return;
      }
    }

    // Clear previous replay state
    cy.nodes().removeClass('replay-current');
    cy.edges().removeClass('replay-edge-active');

    // Apply visited state
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

      // Smart pan
      const ext = cy.extent();
      const pos = currentNode.position();
      if (pos.x < ext.x1 || pos.x > ext.x2 || pos.y < ext.y1 || pos.y > ext.y2) {
        cy.animate({ center: { eles: currentNode }, duration: 150 });
      }
    } else if (replayExpandMode === 'keep') {
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

  const handleSearch = useCallback((term: string) => {
    setSearch(term);
    const cy = cyRef.current;
    if (!cy || !term) return;
    const match = cy.nodes().filter(n => n.data('id')?.includes(term));
    if (match.length) {
      cy.animate({ fit: { eles: match, padding: 50 }, duration: 300 });
    }
  }, []);

  const handleCollapseExpandAll = () => {
    setSelectedNode(null);
    onSelectNode('');
    fitAfterLayoutRef.current = true;
    setCollapsed(allCollapsed ? new Set() : new Set(allLayers));
  };

  const handleFitAll = () => {
    cyRef.current?.animate({ fit: { eles: cyRef.current.elements(), padding: 30 }, duration: 300 });
  };

  const handleCenterSelection = () => {
    const cy = cyRef.current;
    if (!cy) return;
    const highlighted = cy.nodes('.highlighted');
    if (highlighted.length) {
      cy.animate({ center: { eles: highlighted }, duration: 200 });
    } else {
      handleFitAll();
    }
  };

  const handleZoomIn = () => {
    const cy = cyRef.current;
    if (!cy) return;
    const center = { x: cy.width() / 2, y: cy.height() / 2 };
    cy.zoom({ level: cy.zoom() * 1.4, renderedPosition: center });
  };

  const handleZoomOut = () => {
    const cy = cyRef.current;
    if (!cy) return;
    const center = { x: cy.width() / 2, y: cy.height() / 2 };
    cy.zoom({ level: cy.zoom() / 1.4, renderedPosition: center });
  };

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <div className="flex items-center gap-2 px-3 py-2 border-b bg-white shrink-0 flex-wrap text-xs">
        {/* Group: Color Mode */}
        <div className="flex items-center gap-1.5 bg-gray-50 rounded-lg px-2 py-1 border border-gray-200/80">
          <span className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold">Color</span>
          <ColorToggle
            mode={colorMode}
            onChange={setColorMode}
            visibleOps={[...new Set(data.dag.nodes.map(n => n.op))]}
          />
        </div>

        {/* Group: Layout */}
        <div className="flex items-center gap-1 bg-gray-50 rounded-lg px-2 py-1 border border-gray-200/80">
          <span className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold mr-0.5">Layout</span>
          {LAYOUTS.map(l => (
            <button
              key={l.name}
              className={`px-2 py-1 text-xs rounded-md whitespace-nowrap transition-colors font-medium border ${
                layoutName === l.name
                  ? 'bg-indigo-600 text-white border-indigo-600 shadow-sm'
                  : 'bg-white text-gray-600 border-gray-300 hover:bg-gray-100 hover:border-gray-400'
              }`}
              onClick={() => { fitAfterLayoutRef.current = true; setLayoutName(l.name); }}
              title={`${l.label} layout`}
            >{l.label}</button>
          ))}
        </div>

        {/* Group: Layers */}
        <div className="flex items-center gap-1 bg-gray-50 rounded-lg px-2 py-1 border border-gray-200/80">
          <span className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold mr-0.5">Layers</span>
          <button
            className="px-2.5 py-1 text-xs rounded-md whitespace-nowrap transition-colors font-medium border bg-white text-gray-600 border-gray-300 hover:bg-gray-100 hover:border-gray-400"
            onClick={handleCollapseExpandAll}
            title={allCollapsed ? 'Expand all layers' : 'Collapse all layers'}
          >{allCollapsed ? 'Expand All' : 'Collapse All'}</button>
        </div>

        {/* Group: View Controls */}
        <div className="flex items-center gap-1 bg-gray-50 rounded-lg px-2 py-1 border border-gray-200/80">
          <span className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold mr-0.5">View</span>
          {[
            { label: 'Fit', onClick: handleFitAll, title: 'Fit entire graph' },
            { label: 'Center', onClick: handleCenterSelection, title: 'Center on selected' },
            { label: '+', onClick: handleZoomIn, title: 'Zoom in' },
            { label: '\u2013', onClick: handleZoomOut, title: 'Zoom out' },
          ].map(btn => (
            <button
              key={btn.label}
              className="px-2 py-1 text-xs rounded-md whitespace-nowrap transition-colors font-medium border bg-white text-gray-600 border-gray-300 hover:bg-gray-100 hover:border-gray-400"
              onClick={btn.onClick}
              title={btn.title}
            >{btn.label}</button>
          ))}
        </div>

        {/* Search */}
        <input
          type="text"
          placeholder="Search tensor..."
          value={search}
          onChange={e => handleSearch(e.target.value)}
          className="border border-gray-300 rounded-md px-2.5 py-1.5 text-xs w-40 focus:border-indigo-400 focus:ring-1 focus:ring-indigo-200 outline-none"
        />

        {/* Replay */}
        {!replayState && onReplayActivate && (
          <button
            className="px-3 py-1.5 text-xs rounded-md bg-indigo-600 text-white font-medium hover:bg-indigo-700 shadow-sm transition-colors"
            onClick={onReplayActivate}
            title="Replay execution"
          >Replay</button>
        )}
      </div>
      <div className="relative" style={{ flex: '1 1 0', minHeight: 0 }}>
        <div ref={containerRef} style={{ position: 'absolute', inset: 0 }} />
        {colorMode === 'heatmap' && (
          <HeatmapHistogram durations={data.dag.nodes.map(n => n.ns)} />
        )}
      </div>
      {selectedNode && (
        <div className="shrink-0 bg-white border-t border-indigo-100 shadow-lg p-3 z-10">
          <NodeDetail node={selectedNode} onClose={() => setSelectedNode(null)} />
        </div>
      )}
    </div>
  );
}

import { useRef, useEffect, useState, useCallback } from 'react';
import cytoscape, { type Core } from 'cytoscape';
import fcose from 'cytoscape-fcose';
import type { SummaryData, DagNode } from '../types/trace';
import { buildCytoscapeElements, type ColorMode } from '../utils/dagLayout';
import { computeTreeCompoundPositions } from '../utils/treeCompoundLayout';
import { ColorToggle } from './ColorToggle';
import { NodeDetail } from './NodeDetail';
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

  const Btn = ({ onClick, title, children, active }: {
    onClick: () => void; title: string; children: React.ReactNode; active?: boolean;
  }) => (
    <button
      className={`px-2 py-1 text-xs rounded hover:bg-gray-300 whitespace-nowrap
        ${active ? 'bg-gray-800 text-white hover:bg-gray-700' : 'bg-gray-200'}`}
      onClick={onClick}
      title={title}
    >{children}</button>
  );

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <div className="flex items-center gap-1.5 px-2 py-1.5 border-b bg-white shrink-0 flex-wrap text-xs">
        <ColorToggle
          mode={colorMode}
          onChange={setColorMode}
          visibleOps={[...new Set(data.dag.nodes.map(n => n.op))]}
        />

        <div className="w-px h-5 bg-gray-300 mx-0.5" />

        <span className="text-gray-500">Layout:</span>
        {LAYOUTS.map(l => (
          <Btn
            key={l.name}
            active={layoutName === l.name}
            onClick={() => { fitAfterLayoutRef.current = true; setLayoutName(l.name); }}
            title={`${l.label} layout`}
          >{l.label}</Btn>
        ))}

        <div className="w-px h-5 bg-gray-300 mx-0.5" />

        <Btn onClick={handleCollapseExpandAll} title={allCollapsed ? 'Expand all layers' : 'Collapse all layers'}>
          {allCollapsed ? '⊞ Expand' : '⊟ Collapse'}
        </Btn>

        <div className="w-px h-5 bg-gray-300 mx-0.5" />

        <Btn onClick={handleFitAll} title="Fit entire graph in view">⊡ Fit</Btn>
        <Btn onClick={handleCenterSelection} title="Center on selected node">◎ Center</Btn>
        <Btn onClick={handleZoomIn} title="Zoom in">＋</Btn>
        <Btn onClick={handleZoomOut} title="Zoom out">－</Btn>

        <div className="w-px h-5 bg-gray-300 mx-0.5" />

        <span className="text-gray-500">🔍</span>
        <input
          type="text"
          placeholder="Search tensor..."
          value={search}
          onChange={e => handleSearch(e.target.value)}
          className="border rounded px-2 py-1 text-xs w-36"
        />

        {!replayState && onReplayActivate && (
          <>
            <div className="w-px h-5 bg-gray-300 mx-0.5" />
            <Btn onClick={onReplayActivate} title="Replay execution">
              Replay
            </Btn>
          </>
        )}
      </div>
      <div ref={containerRef} style={{ flex: '1 1 0', minHeight: 0 }} />
      {selectedNode && (
        <div className="shrink-0 bg-white border-t shadow-lg p-3 z-10">
          <NodeDetail node={selectedNode} onClose={() => setSelectedNode(null)} />
        </div>
      )}
    </div>
  );
}

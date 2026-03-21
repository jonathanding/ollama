import { useRef, useEffect, useState, useCallback } from 'react';
import cytoscape, { type Core } from 'cytoscape';
import type { SummaryData, DagNode } from '../types/trace';
import { buildCytoscapeElements, type ColorMode } from '../utils/dagLayout';
import { ColorToggle } from './ColorToggle';
import { NodeDetail } from './NodeDetail';

interface Props {
  data: SummaryData;
  highlightId: string | null;
  onSelectNode: (id: string) => void;
}

export function DagView({ data, highlightId, onSelectNode }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const [colorMode, setColorMode] = useState<ColorMode>('backend');
  const [collapsed, setCollapsed] = useState<Set<string>>(() => {
    const layers = new Set<string>();
    for (const n of data.dag.nodes) { if (n.layer) layers.add(n.layer); }
    return layers;
  });
  const [selectedNode, setSelectedNode] = useState<DagNode | null>(null);
  const [search, setSearch] = useState('');

  useEffect(() => {
    if (!containerRef.current) return;
    const elements = buildCytoscapeElements(data.dag, colorMode, collapsed);
    const cy = cytoscape({
      container: containerRef.current,
      elements,
      layout: { name: 'breadthfirst', directed: true, spacingFactor: 1.5 },
      style: [
        { selector: 'node', style: {
          label: 'data(label)', 'font-size': 10, 'text-wrap': 'wrap',
          'text-valign': 'center', 'text-halign': 'center',
        }},
        { selector: ':parent', style: {
          'text-valign': 'top', 'font-weight': 'bold',
        }},
        { selector: 'edge', style: {
          'curve-style': 'bezier', 'target-arrow-shape': 'triangle',
          'arrow-scale': 0.8, 'line-color': '#9ca3af', 'target-arrow-color': '#9ca3af',
        }},
        { selector: '.highlighted', style: {
          'border-color': '#2563eb', 'border-width': 4,
          'overlay-color': '#2563eb', 'overlay-opacity': 0.15,
        }},
      ],
    });

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

    cyRef.current = cy;
    return () => { cy.destroy(); };
  }, [data.dag, colorMode, collapsed]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !highlightId) return;
    cy.nodes().removeClass('highlighted');
    const node = cy.getElementById(highlightId);
    if (node.length) {
      node.addClass('highlighted');
      cy.animate({ center: { eles: node }, duration: 300 });
    }
  }, [highlightId]);

  const handleSearch = useCallback((term: string) => {
    setSearch(term);
    const cy = cyRef.current;
    if (!cy || !term) return;
    const match = cy.nodes().filter(n => n.data('id')?.includes(term));
    if (match.length) {
      cy.animate({ fit: { eles: match, padding: 50 }, duration: 300 });
    }
  }, []);

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-4 p-2 border-b">
        <ColorToggle
          mode={colorMode}
          onChange={setColorMode}
          visibleOps={[...new Set(data.dag.nodes.map(n => n.op))]}
        />
        <input
          type="text"
          placeholder="Search tensor..."
          value={search}
          onChange={e => handleSearch(e.target.value)}
          className="border rounded px-2 py-1 text-sm flex-1 max-w-xs"
        />
      </div>
      <div ref={containerRef} className="flex-1" />
      <NodeDetail node={selectedNode} onClose={() => setSelectedNode(null)} />
    </div>
  );
}

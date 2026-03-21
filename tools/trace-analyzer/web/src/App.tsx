import { useState, useEffect } from 'react';
import { useFileList, useSummary, useCompare } from './hooks/useTraceData';
import { DagView } from './components/DagView';
import { TimelineView } from './components/TimelineView';
import { CompareView } from './components/CompareView';
import { HotspotPanel } from './components/HotspotPanel';
import { ReplayPanel, type ReplayState, type ExpandMode } from './components/ReplayPanel';

type View = 'dag' | 'timeline' | 'compare';

export default function App() {
  const files = useFileList();
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [compareFile, setCompareFile] = useState<string | null>(null);
  const [view, setView] = useState<View>('dag');
  const [highlightId, setHighlightId] = useState<string | null>(null);
  const [replayActive, setReplayActive] = useState(false);
  const [replayState, setReplayState] = useState<ReplayState | null>(null);
  const [replayExpandMode, setReplayExpandMode] = useState<ExpandMode>('keep');

  const summaryFiles = files.filter(f => f.type === 'summary');
  const compareFiles = files.filter(f => f.type === 'compare');

  const { data: summaryData } = useSummary(
    view !== 'compare' ? selectedFile : null
  );
  const { data: compareData } = useCompare(
    view === 'compare' ? compareFile : null
  );

  const handleReplayActivate = () => {
    setReplayActive(true);
    setHighlightId(null);
  };

  useEffect(() => {
    if (view !== 'dag') {
      setReplayActive(false);
      setReplayState(null);
    }
  }, [view]);

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      <div className="bg-indigo-700 px-4 py-2.5 flex items-center gap-4 shrink-0 shadow-md">
        <h1 className="font-bold text-lg whitespace-nowrap text-white tracking-tight">Trace Analyzer</h1>
        <div className="flex gap-0.5 bg-indigo-800/60 rounded-lg p-0.5">
          {(['dag', 'timeline', 'compare'] as View[]).map(v => (
            <button
              key={v}
              className={`px-3 py-1.5 rounded-md text-sm capitalize transition-colors ${view === v ? 'bg-white text-indigo-700 shadow font-semibold' : 'text-indigo-200 hover:text-white hover:bg-indigo-600/50'}`}
              onClick={() => setView(v)}
            >{v === 'dag' ? 'DAG' : v}</button>
          ))}
        </div>
        {view !== 'compare' ? (
          <select
            className={`rounded-md px-2.5 py-1.5 text-sm outline-none font-medium transition-colors [&>option]:text-gray-900 [&>option]:bg-white ${
              selectedFile
                ? 'bg-amber-500 text-white border border-amber-400 focus:ring-2 focus:ring-amber-200'
                : 'bg-amber-400 text-amber-950 border border-amber-300 focus:ring-2 focus:ring-amber-200 animate-pulse'
            }`}
            value={selectedFile ?? ''}
            onChange={e => setSelectedFile(e.target.value || null)}
          >
            <option value="">Select summary...</option>
            {summaryFiles.map(f => <option key={f.name} value={f.name}>{f.name}</option>)}
          </select>
        ) : (
          <select
            className={`rounded-md px-2.5 py-1.5 text-sm outline-none font-medium transition-colors [&>option]:text-gray-900 [&>option]:bg-white ${
              compareFile
                ? 'bg-amber-500 text-white border border-amber-400 focus:ring-2 focus:ring-amber-200'
                : 'bg-amber-400 text-amber-950 border border-amber-300 focus:ring-2 focus:ring-amber-200 animate-pulse'
            }`}
            value={compareFile ?? ''}
            onChange={e => setCompareFile(e.target.value || null)}
          >
            <option value="">Select compare...</option>
            {compareFiles.map(f => <option key={f.name} value={f.name}>{f.name}</option>)}
          </select>
        )}
      </div>

      <div className="flex-1 flex overflow-hidden min-h-0">
        <div className="flex-1 flex flex-col min-w-0 min-h-0 overflow-hidden">
          {view === 'dag' && summaryData && (
            <DagView
              data={summaryData}
              highlightId={highlightId}
              onSelectNode={setHighlightId}
              replayState={replayState}
              replayExpandMode={replayExpandMode}
              onReplayActivate={handleReplayActivate}
            />
          )}
          {view === 'timeline' && summaryData && (
            <TimelineView data={summaryData} onSelectOp={setHighlightId} />
          )}
          {view === 'compare' && compareData && (
            <CompareView data={compareData} />
          )}
          {!summaryData && view !== 'compare' && (
            <div className="flex-1 flex items-center justify-center text-gray-400">Select a summary file to begin</div>
          )}
          {!compareData && view === 'compare' && (
            <div className="flex-1 flex items-center justify-center text-gray-400">Select a compare file to begin</div>
          )}
        </div>
        {summaryData && view !== 'compare' && (
          replayActive && view === 'dag' ? (
            <ReplayPanel
              data={summaryData}
              onReplayState={setReplayState}
              onStop={() => { setReplayActive(false); setReplayState(null); }}
              expandMode={replayExpandMode}
              onExpandModeChange={setReplayExpandMode}
            />
          ) : (
            <HotspotPanel data={summaryData} selectedId={highlightId} onSelect={setHighlightId} />
          )
        )}
      </div>
    </div>
  );
}

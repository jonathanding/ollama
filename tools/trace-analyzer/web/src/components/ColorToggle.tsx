import type { ColorMode } from '../utils/dagLayout';

interface Props {
  mode: ColorMode;
  onChange: (mode: ColorMode) => void;
}

export function ColorToggle({ mode, onChange }: Props) {
  return (
    <div className="flex gap-1 rounded-lg bg-gray-100 p-1">
      <button
        className={`px-3 py-1 rounded text-sm ${mode === 'backend' ? 'bg-white shadow' : ''}`}
        onClick={() => onChange('backend')}
      >Backend</button>
      <button
        className={`px-3 py-1 rounded text-sm ${mode === 'heatmap' ? 'bg-white shadow' : ''}`}
        onClick={() => onChange('heatmap')}
      >Heatmap</button>
    </div>
  );
}

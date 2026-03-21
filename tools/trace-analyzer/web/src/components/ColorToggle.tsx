import type { ColorMode } from '../utils/dagLayout';
import { BACKEND_COLORS, OP_COLORS } from '../utils/colorScale';

interface Props {
  mode: ColorMode;
  onChange: (mode: ColorMode) => void;
  visibleOps?: string[];
}

export function ColorToggle({ mode, onChange, visibleOps }: Props) {
  // Deduplicate by color value, keep first key per color
  const dedup = (entries: [string, string][]) => {
    const seen = new Set<string>();
    return entries.filter(([, color]) => {
      if (seen.has(color)) return false;
      seen.add(color);
      return true;
    });
  };

  const legendItems =
    mode === 'backend'
      ? dedup(Object.entries(BACKEND_COLORS))
      : mode === 'op'
      ? Object.entries(OP_COLORS).filter(
          ([op]) => !visibleOps || visibleOps.includes(op)
        )
      : [];

  return (
    <div className="flex items-center gap-3 flex-wrap">
      <div className="flex gap-0.5 rounded-lg bg-gray-100 p-0.5">
        {(['backend', 'op', 'heatmap'] as ColorMode[]).map(m => (
          <button
            key={m}
            className={`px-2 py-1 rounded text-xs ${mode === m ? 'bg-white shadow font-medium' : 'text-gray-600'}`}
            onClick={() => onChange(m)}
          >{m === 'op' ? 'Op Type' : m === 'backend' ? 'Backend' : 'Heatmap'}</button>
        ))}
      </div>
      {legendItems.length > 0 && (
        <div className="flex flex-wrap gap-x-2 gap-y-0.5 text-xs text-gray-600">
          {legendItems.map(([label, color]) => (
            <span key={label} className="flex items-center gap-1">
              <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: color }} />
              {label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

import type { ColorMode } from '../utils/dagLayout';
import { BACKEND_COLORS, OP_COLORS } from '../utils/colorScale';

interface Props {
  mode: ColorMode;
  onChange: (mode: ColorMode) => void;
  visibleOps?: string[];
}

export function ColorToggle({ mode, onChange, visibleOps }: Props) {
  const legendItems =
    mode === 'backend'
      ? Object.entries(BACKEND_COLORS).filter(([, v], i, arr) =>
          arr.findIndex(([, v2]) => v2 === v) === i
        )
      : mode === 'op'
      ? Object.entries(OP_COLORS).filter(
          ([op]) => !visibleOps || visibleOps.includes(op)
        )
      : [];

  return (
    <div className="flex items-center gap-3">
      <div className="flex gap-1 rounded-lg bg-gray-100 p-1">
        {(['backend', 'op', 'heatmap'] as ColorMode[]).map(m => (
          <button
            key={m}
            className={`px-3 py-1 rounded text-sm capitalize ${mode === m ? 'bg-white shadow' : ''}`}
            onClick={() => onChange(m)}
          >{m === 'op' ? 'Op Type' : m === 'backend' ? 'Backend' : 'Heatmap'}</button>
        ))}
      </div>
      {legendItems.length > 0 && (
        <div className="flex flex-wrap gap-2 text-xs">
          {legendItems.slice(0, 12).map(([label, color]) => (
            <span key={label} className="flex items-center gap-1">
              <span
                className="inline-block w-3 h-3 rounded"
                style={{ backgroundColor: color }}
              />
              {label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

'use client';

import { Prediction } from '@/lib/types';

interface Props {
  predictions: Prediction[];
}

const REGIME_LABELS: Record<string, string> = {
  goldilocks:             'Goldilocks',
  reflation:              'Reflation',
  overheating:            'Overheating',
  stagflation:            'Stagflation',
  transition:             'Transition',
  secular_stagnation:     'Secular Stagnation',
  deflationary_recession: 'Deflationary Recession',
};

const REGIME_COLORS: Record<string, string> = {
  goldilocks:             '#2ecc71',
  reflation:              '#e67e22',
  overheating:            '#e74c3c',
  stagflation:            '#8e44ad',
  transition:             '#f59e0b',
  secular_stagnation:     '#3b82f6',
  deflationary_recession: '#475569',
};

export default function SignalCard({ predictions }: Props) {
  const latest = predictions[predictions.length - 1];
  const isHawkish = latest.predicted > 0.05;
  const isDovish  = latest.predicted < -0.05;
  const direction = isHawkish ? 'Higher' : isDovish ? 'Lower' : 'Flat';
  const dirColor  = isHawkish ? '#ef4444' : isDovish ? '#22c55e' : '#f59e0b';
  const dirArrow  = isHawkish ? '↑' : isDovish ? '↓' : '→';

  return (
    <div className="rounded-xl border border-[#1e2d50] bg-[#0f1629] p-6">
      <p className="text-xs font-semibold tracking-widest text-[#64748b] uppercase mb-4">
        Latest Signal
      </p>

      <div className="flex items-start justify-between gap-4 mb-6">
        <div>
          <p className="text-sm text-[#64748b] mb-1">{latest.label}</p>
          <p style={{ color: dirColor }} className="text-5xl font-bold tracking-tight">
            {dirArrow} {direction}
          </p>
          <p className="text-[#94a3b8] text-sm mt-2">
            Predicted 1Y yield change:{' '}
            <span className="font-mono font-semibold" style={{ color: dirColor }}>
              {latest.predicted > 0 ? '+' : ''}{latest.predicted.toFixed(2)}%
            </span>
          </p>
        </div>

        <div
          className="rounded-lg px-3 py-1.5 text-xs font-semibold shrink-0"
          style={{
            background: `${REGIME_COLORS[latest.regime]}22`,
            color: REGIME_COLORS[latest.regime],
            border: `1px solid ${REGIME_COLORS[latest.regime]}55`,
          }}
        >
          {REGIME_LABELS[latest.regime] ?? latest.regime}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4 pt-4 border-t border-[#1e2d50] text-center">
        {predictions
          .filter(p => p.actual !== null)
          .slice(-3)
          .map(p => {
            const correct = Math.sign(p.predicted) === Math.sign(p.actual!);
            return (
              <div key={p.date}>
                <p className="text-xs text-[#64748b] mb-1">{p.label}</p>
                <p
                  className="text-sm font-semibold font-mono"
                  style={{ color: correct ? '#22c55e' : '#ef4444' }}
                >
                  {correct ? '✓' : '✗'}{' '}
                  {p.predicted > 0 ? '+' : ''}{p.predicted.toFixed(2)}
                </p>
              </div>
            );
          })}
      </div>
    </div>
  );
}

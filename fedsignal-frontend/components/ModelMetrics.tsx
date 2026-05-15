'use client';

import { Metrics } from '@/lib/types';

interface Props {
  metrics: Metrics;
}

const FEATURE_LABELS: Record<string, string> = {
  twoy_ff_spread:       '2Y − FF Spread',
  sent_level_demeaned:  'NLP Hawkishness',
  nlp_vs_regime:        'NLP vs Regime',
  core_pce_yoy:         'Core PCE YoY',
  regime_ordinal:       'Regime (Ordinal)',
  nlp_momentum:         'NLP Momentum',
  sent_dispersion:      'Sent. Dispersion',
};

export default function ModelMetrics({ metrics }: Props) {
  const topImportance = metrics.feature_importances[0].importance;

  return (
    <div className="rounded-xl border border-[#1e2d50] bg-[#0f1629] p-6">
      <p className="text-xs font-semibold tracking-widest text-[#64748b] uppercase mb-1">
        Model Performance
      </p>
      <p className="text-white font-semibold mb-6">GBR · Walk-Forward CV · {metrics.n_test} OOS Predictions</p>

      <div className="grid grid-cols-3 gap-4 mb-8">
        {[
          { label: 'OOS R²',     value: (metrics.oos_r2 > 0 ? '+' : '') + metrics.oos_r2.toFixed(3), color: metrics.oos_r2 > 0 ? '#22c55e' : '#ef4444' },
          { label: 'Dir. Acc.', value: metrics.dir_acc.toFixed(1) + '%',  color: '#3b82f6' },
          { label: 'MAE',        value: metrics.mae.toFixed(3) + '%',     color: '#94a3b8' },
        ].map(stat => (
          <div key={stat.label} className="rounded-lg bg-[#162040] border border-[#1e2d50] p-4 text-center">
            <p className="text-xs text-[#64748b] mb-1">{stat.label}</p>
            <p className="text-2xl font-bold font-mono" style={{ color: stat.color }}>
              {stat.value}
            </p>
          </div>
        ))}
      </div>

      <p className="text-xs font-semibold tracking-widest text-[#64748b] uppercase mb-4">
        Feature Importance (Gini)
      </p>
      <div className="space-y-3">
        {metrics.feature_importances.map(({ feature, importance }) => (
          <div key={feature} className="flex items-center gap-3">
            <p className="text-xs text-[#94a3b8] w-36 shrink-0">
              {FEATURE_LABELS[feature] ?? feature}
            </p>
            <div className="flex-1 h-2 rounded-full bg-[#162040] overflow-hidden">
              <div
                className="h-full rounded-full bg-[#3b82f6] transition-all duration-500"
                style={{ width: `${(importance / topImportance) * 100}%` }}
              />
            </div>
            <p className="text-xs font-mono text-[#64748b] w-10 text-right">
              {(importance * 100).toFixed(1)}%
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

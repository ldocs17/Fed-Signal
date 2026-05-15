'use client';

import {
  ComposedChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ReferenceLine, ResponsiveContainer,
  ReferenceArea,
} from 'recharts';
import { Prediction } from '@/lib/types';

interface Props {
  data: Prediction[];
}

const REGIME_COLORS: Record<string, string> = {
  goldilocks:             '#2ecc71',
  reflation:              '#e67e22',
  overheating:            '#e74c3c',
  stagflation:            '#8e44ad',
  transition:             '#f59e0b',
  secular_stagnation:     '#3b82f6',
  deflationary_recession: '#475569',
};

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-[#1e2d50] bg-[#0f1629] px-3 py-2 text-sm shadow-xl">
      <p className="text-[#94a3b8] mb-2">{label}</p>
      {payload.map((p: any) => (
        <p key={p.name} className="font-mono" style={{ color: p.color }}>
          {p.name}: {p.value != null ? `${p.value > 0 ? '+' : ''}${Number(p.value).toFixed(3)}%` : '—'}
        </p>
      ))}
    </div>
  );
};

export default function PredVsActualChart({ data }: Props) {
  const withActual = data.filter(d => d.actual !== null);

  return (
    <div className="rounded-xl border border-[#1e2d50] bg-[#0f1629] p-6">
      <p className="text-xs font-semibold tracking-widest text-[#64748b] uppercase mb-1">
        Walk-Forward CV
      </p>
      <p className="text-white font-semibold mb-6">Predicted vs Actual Yield Change</p>

      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={withActual} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
          {withActual.map((d, i) => {
            if (i === withActual.length - 1) return null;
            return (
              <ReferenceArea
                key={d.date}
                x1={d.label}
                x2={withActual[i + 1].label}
                fill={REGIME_COLORS[d.regime] ?? '#334155'}
                fillOpacity={0.08}
              />
            );
          })}
          <CartesianGrid strokeDasharray="3 3" stroke="#1e2d50" vertical={false} />
          <XAxis
            dataKey="label"
            tick={{ fill: '#64748b', fontSize: 10 }}
            interval={3}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: '#64748b', fontSize: 10 }}
            axisLine={false}
            tickLine={false}
            tickFormatter={v => `${v > 0 ? '+' : ''}${v.toFixed(1)}%`}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: '12px', color: '#64748b', paddingTop: '8px' }}
          />
          <ReferenceLine y={0} stroke="#334155" strokeWidth={1} />
          <Line
            type="monotone"
            dataKey="actual"
            name="Actual"
            stroke="#e2e8f0"
            strokeWidth={2}
            dot={false}
            connectNulls
          />
          <Line
            type="monotone"
            dataKey="predicted"
            name="Predicted"
            stroke="#3b82f6"
            strokeWidth={2}
            strokeDasharray="5 3"
            dot={false}
          />
        </ComposedChart>
      </ResponsiveContainer>

      <div className="flex flex-wrap gap-4 mt-4 text-xs text-[#64748b]">
        {Object.entries(REGIME_COLORS).map(([regime, color]) => (
          <span key={regime} className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full" style={{ background: color }} />
            {regime.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
          </span>
        ))}
      </div>
    </div>
  );
}

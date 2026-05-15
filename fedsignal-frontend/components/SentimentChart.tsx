'use client';

import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer, Cell,
} from 'recharts';
import { SentimentPoint } from '@/lib/types';

interface Props {
  data: SentimentPoint[];
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  const val: number = payload[0].value;
  return (
    <div className="rounded-lg border border-[#1e2d50] bg-[#0f1629] px-3 py-2 text-sm shadow-xl">
      <p className="text-[#94a3b8] mb-1">{label}</p>
      <p className="font-semibold" style={{ color: val >= 0 ? '#ef4444' : '#22c55e' }}>
        {val >= 0 ? 'Hawkish' : 'Dovish'}: {val > 0 ? '+' : ''}{val.toFixed(3)}
      </p>
    </div>
  );
};

export default function SentimentChart({ data }: Props) {
  const recent = data.slice(-24);

  return (
    <div className="rounded-xl border border-[#1e2d50] bg-[#0f1629] p-6">
      <p className="text-xs font-semibold tracking-widest text-[#64748b] uppercase mb-1">
        NLP Sentiment
      </p>
      <p className="text-white font-semibold mb-6">FOMC Hawkishness Over Time</p>

      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={recent} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
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
            tickFormatter={v => v.toFixed(2)}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: '#1e2d5044' }} />
          <ReferenceLine y={0} stroke="#334155" strokeWidth={1} />
          <Bar dataKey="score" radius={[2, 2, 0, 0]} maxBarSize={18}>
            {recent.map((entry, i) => (
              <Cell key={i} fill={entry.score >= 0 ? '#ef4444' : '#22c55e'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <div className="flex gap-4 mt-4 text-xs text-[#64748b]">
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-sm bg-[#ef4444]" /> Hawkish
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-sm bg-[#22c55e]" /> Dovish
        </span>
      </div>
    </div>
  );
}

import { readFile } from 'fs/promises';
import path from 'path';

import Hero from '@/components/Hero';
import SignalCard from '@/components/SignalCard';
import SentimentChart from '@/components/SentimentChart';
import PredVsActualChart from '@/components/PredVsActualChart';
import ModelMetrics from '@/components/ModelMetrics';
import StatementDemo from '@/components/StatementDemo';

import type { Prediction, SentimentPoint, Metrics, ExampleStatement } from '@/lib/types';

async function loadJson<T>(filename: string): Promise<T> {
  const file = path.join(process.cwd(), 'public', 'data', filename);
  const raw = await readFile(file, 'utf-8');
  return JSON.parse(raw) as T;
}

export default async function Page() {
  const [predictions, sentiment, metrics, examples] = await Promise.all([
    loadJson<Prediction[]>('predictions.json'),
    loadJson<SentimentPoint[]>('sentiment.json'),
    loadJson<Metrics>('metrics.json'),
    loadJson<ExampleStatement[]>('examples.json'),
  ]);

  return (
    <main className="mx-auto max-w-6xl px-4 py-12 space-y-10">
      <Hero />

      <SignalCard predictions={predictions} />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <SentimentChart data={sentiment} />
        <PredVsActualChart data={predictions} />
      </div>

      <ModelMetrics metrics={metrics} />

      <StatementDemo examples={examples} />

      <footer className="text-center text-xs text-[#475569] pb-4">
        Data sourced from FRED and federalreserve.gov · Model: GBR + DistilBERT fine-tuned on 88 labeled FOMC sentences
      </footer>
    </main>
  );
}

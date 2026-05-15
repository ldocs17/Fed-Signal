import type { Metadata } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });
const mono  = JetBrains_Mono({ subsets: ['latin'], variable: '--font-mono' });

export const metadata: Metadata = {
  title: 'FedSignal — FOMC Rate Prediction',
  description:
    'ML-powered US Treasury yield forecasting using NLP sentiment from FOMC statements and macroeconomic regime classification.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${mono.variable} h-full`}>
      <body className="min-h-full bg-[#080d1a] font-sans antialiased">
        {children}
      </body>
    </html>
  );
}

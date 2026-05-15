'use client';

export default function Hero() {
  return (
    <header className="border-b border-[#1e2d50] bg-[#0f1629]">
      <div className="max-w-6xl mx-auto px-6 py-12">
        <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-6">
          <div>
            <div className="flex items-center gap-3 mb-3">
              <span className="text-xs font-semibold tracking-widest text-[#3b82f6] uppercase">
                Federal Reserve · NLP · Machine Learning
              </span>
            </div>
            <h1 className="text-4xl font-bold text-white mb-3 tracking-tight">
              FedSignal
            </h1>
            <p className="text-[#94a3b8] max-w-xl leading-relaxed">
              Predicts US Treasury yield movements following FOMC meetings by combining
              NLP sentiment analysis of Federal Reserve statements with macroeconomic
              regime classification and gradient boosting.
            </p>
          </div>

          <div className="flex gap-3 shrink-0">
            <a
              href="https://github.com/ldocs17/Fed-Signal"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[#162040] border border-[#1e2d50]
                         text-sm text-[#e2e8f0] hover:border-[#3b82f6] hover:text-white transition-colors"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57
                         0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695
                         -.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99
                         .105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225
                         -.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405
                         c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225
                         0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3
                         0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z" />
              </svg>
              GitHub
            </a>
          </div>
        </div>
      </div>
    </header>
  );
}

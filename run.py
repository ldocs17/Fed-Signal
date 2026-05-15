import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch

from fedsignal.data.fred_loader import load_fred_data, build_regime_data
from fedsignal.data.fomc_scraper import load_fomc_statements, build_fomc_df
from fedsignal.models.nlp import train_nlp_model, score_fomc_df
from fedsignal.models.gbr import engineer_features, run_walk_forward, print_metrics
from fedsignal.visualize import plot_results


def main():
    random.seed(24)
    np.random.seed(24)
    torch.manual_seed(24)

    # 1. FRED data + regime classification
    fred_cache     = load_fred_data()
    regime_monthly = build_regime_data(fred_cache)

    # 2. FOMC statements + target variable
    statements = load_fomc_statements()
    fomc_df    = build_fomc_df(statements, regime_monthly, fred_cache['one_year_daily'])

    # 3. Fine-tune DistilBERT + score statements
    nlp_model, tokenizer = train_nlp_model()
    fomc_df = score_fomc_df(fomc_df, nlp_model, tokenizer)

    # 4. Feature engineering
    fomc_df = engineer_features(fomc_df)

    # 5. GBR walk-forward evaluation
    results = run_walk_forward(fomc_df)
    print_metrics(results)

    # 6. Visualization
    plot_results(fomc_df, results)


if __name__ == '__main__':
    main()

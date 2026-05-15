import matplotlib.pyplot as plt

REGIME_COLORS = {
    'goldilocks':             '#2ecc71',
    'reflation':              '#e67e22',
    'overheating':            '#e74c3c',
    'stagflation':            '#8e44ad',
    'transition':             '#f1c40f',
    'secular_stagnation':     '#3498db',
    'deflationary_recession': '#2c3e50',
}


def plot_results(fomc_df, results):
    preds        = results['preds']
    acts         = results['acts']
    oos_r2       = results['oos_r2']
    dir_acc      = results['dir_acc']
    test_dates   = results['test_dates']
    test_regimes = results['test_regimes']

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('newFedSignal: US Federal Reserve (GBR + Walk-Forward)', fontsize=14, fontweight='bold')

    valid_us = fomc_df['sent_level_demeaned'].dropna()
    ax0.bar(valid_us.index, valid_us.values, width=25,
            color='#3498db', edgecolor='black', linewidth=0.3, alpha=0.8)
    ax0.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax0.set_title('FOMC NLP Sentiment (Demeaned)', fontsize=11)
    ax0.set_ylabel('NLP Score')

    ax1.plot(test_dates, acts,  'k-', linewidth=1.8, label='Actual',    zorder=3)
    ax1.plot(test_dates, preds, 'b-', linewidth=1.5, label='Predicted', alpha=0.8)
    for i in range(len(test_dates) - 1):
        ax1.axvspan(test_dates[i], test_dates[i + 1], alpha=0.15,
                    color=REGIME_COLORS.get(test_regimes[i], '#bdc3c7'))
    ax1.axhline(0, color='red', linestyle='--', linewidth=0.8)
    ax1.set_title(f'GBR Pred vs Actual  (OOS R²={oos_r2:+.3f}, Dir={dir_acc:.1f}%)', fontsize=11)
    ax1.set_ylabel('1Y Yield 60d Change')
    ax1.legend(fontsize=9)

    plt.tight_layout()
    plt.show()

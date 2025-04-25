import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis, probplot
from statsmodels.graphics.tsaplots import plot_acf

def visualize_return(mid_prices, plotTitle=None):
    """
    Analyze the return series based on mid-price and visualize:
    1. Mid-price series.
    2. Autocorrelation of returns.
    3. Distribution of returns with normal distribution fit.
    4. Q-Q plot of returns against normal distribution.
    5. Autocorrelation of squared returns (volatility clustering).
    6. Autocorrelation of absolute returns (volatility clustering).

    Parameters:
    - mid_prices (array-like): Time series of mid prices.
    - plotTitle (str, optional): Title for the overall plot. Default is None.
    """
    # Step 1: Calculate log returns: ln(mid_price_t+1) - ln(mid_price_t)
    log_mid_prices = np.log(mid_prices)
    returns = np.diff(log_mid_prices)  # This calculates the difference between each time step

    # Step 2: Calculate sample statistics for returns (used as MLE for normal fit)
    mean_return = np.mean(returns)
    variance_return = np.var(returns)
    std_return = np.sqrt(variance_return)  # Standard deviation for the normal fit
    skewness_return = skew(returns)
    kurtosis_return = kurtosis(returns)

    # Step 3: Calculate squared and absolute returns for volatility clustering analysis
    squared_returns = returns ** 2
    absolute_returns = np.abs(returns)

    # Step 4: Set up a 3x2 plot grid
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    # Set the main title if plotTitle is provided
    if plotTitle:
        fig.suptitle(plotTitle, fontsize=16)
        plt.subplots_adjust(top=0.92)  # Adjust to make space for the title

    # Plot 1: Mid-price series
    axes[0, 0].plot(mid_prices, color='blue')
    axes[0, 0].set_title("Mid-Price Series")
    axes[0, 0].set_xlabel("Time Step")
    axes[0, 0].set_ylabel("Mid Price")

    # Plot 2: Autocorrelation of returns
    plot_acf(returns, lags=30, ax=axes[0, 1])
    axes[0, 1].set_title("Autocorrelation of Returns")
    axes[0, 1].set_xlabel("Lag")
    axes[0, 1].set_ylabel("Autocorrelation")

    # Plot 3: Distribution of returns with sample statistics and normal fit
    n, bins, _ = axes[1, 0].hist(returns, bins=50, alpha=0.7, color='blue', density=True)
    axes[1, 0].set_title("Distribution of Returns")
    axes[1, 0].set_xlabel("Return")
    axes[1, 0].set_ylabel("Density")

    # Fit a normal distribution using MLE estimates
    x = np.linspace(min(bins), max(bins), 100)
    fitted_pdf = norm.pdf(x, mean_return, std_return)
    axes[1, 0].plot(x, fitted_pdf, color='red', linestyle='--', linewidth=2, label='Normal Fit')

    # Display sample statistics on the histogram plot
    stats_text = (
        f"Mean: {mean_return:.6f}\n"
        f"Variance: {variance_return:.6f}\n"
        f"Skewness: {skewness_return:.6f}\n"
        f"Excess Kurtosis: {kurtosis_return:.6f}"
    )
    axes[1, 0].text(0.98, 0.95, stats_text, transform=axes[1, 0].transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    # Plot 4: Q-Q plot of returns against normal distribution to check for heavy tails
    probplot(returns, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q Plot of Returns vs Normal")
    axes[1, 1].set_xlabel("Theoretical Quantiles")
    axes[1, 1].set_ylabel("Sample Quantiles")

    # Plot 5: Autocorrelation of squared returns (volatility clustering)
    plot_acf(squared_returns, lags=30, ax=axes[2, 0])
    axes[2, 0].set_title("Autocorrelation of Squared Returns")
    axes[2, 0].set_xlabel("Lag")
    axes[2, 0].set_ylabel("Autocorrelation")

    # Plot 6: Autocorrelation of absolute returns (alternative measure of volatility clustering)
    plot_acf(absolute_returns, lags=30, ax=axes[2, 1])
    axes[2, 1].set_title("Autocorrelation of Absolute Returns")
    axes[2, 1].set_xlabel("Lag")
    axes[2, 1].set_ylabel("Autocorrelation") 

    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()

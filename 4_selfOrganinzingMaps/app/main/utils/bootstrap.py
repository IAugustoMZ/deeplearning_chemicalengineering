import numpy as np
from typing import List
import matplotlib.pyplot as plt

def bootstrap(data, n_bootstraps=1000):
    """
    calculates the bootstrapped mean and 95% confidence interval

    :param data: data to be bootstrapped
    :type data: pd.Series or pd.DataFrame
    :param n_bootstraps: number of resamples, defaults to 1000
    :type n_bootstraps: int, optional
    """
    # convert to numpy array
    data = data.values

    # calculate the bootstrapped mean
    bootstrapped_means = np.array([
        np.mean(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstraps)
    ])

    return bootstrapped_means

# create a function to plot the distributions of each variable
def plot_bootstrap_samples(boot_dict, col):
    """
    plots the bootstrapped samples for each cluster

    :param boot_dict: dictionary with the bootstrapped samples
    :type boot_dict: dict
    :param col: column to be plotted
    :type col: str
    """
    # create the figure
    plt.figure(figsize=(10, 5))

    # plot the bootstrapped samples
    for c in boot_dict[col].keys():

        # calculate the 95% confidence interval
        lower = round(np.percentile(boot_dict[col][c], 2.5), 2)
        upper = round(np.percentile(boot_dict[col][c], 97.5), 2)

        # plot the histogram
        plt.hist(boot_dict[col][c], bins=30, alpha=0.5, label=f'Cluster {c} - ({lower}-{upper})')

        # plot the confidence interval
        plt.axvline(lower, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(upper, color='gray', linestyle='--', alpha=0.5)

    # add the legend
    plt.legend()

    # add the title
    plt.title(f'Bootstrapped Samples - {col}', size=16)

    # show the plot
    plt.show()

# create a function to plot multiple histograms
def plot_histograms(list_data: List[dict]):
    """
    plots multiple histograms in a single plot

    :param list_data: list of dictionaries with the data
    :type list_data: List[dict]
    """
    # create the figure
    plt.figure(figsize=(10, 5))

    # plot the histograms
    for data in list_data:
        plt.hist(data['data'], bins=30, alpha=0.5, label=data['label'])

    # add the legend
    plt.legend()

    # add the title
    plt.title('Histograms', size=16)

    # show the plot
    plt.show()
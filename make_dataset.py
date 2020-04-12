# Leonard R. Kosta Jr.

import argparse
import numpy as np
import scipy.stats
import random
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULT_NUM_EXAMPLES = 1000
DEFAULT_NUM_FEATURES = 2
DEFAULT_OUTFILE = 'data_test.hdf5'
DEFAULT_PLOT_FILE = 'data_plot_test.png'
DEFAULT_DATA_KEY = 'data'
DEFAULT_LABEL_KEY = 'labels'
FUNCTION_UNIT_CIRCLE = 0
FUNCTION_FEATURES_POSITIVE = 1


def get_random_points(dim):
    """Returns a numpy array of dim dimensions with Gaussian
    distributed points over [-1, 1)."""
    return np.random.random_sample(dim) * 2 - 1


def get_labels(X, function=None):
    """Returns a numpy array of labels for the points in X. The labels
    are determined by applying function to each example in X.""" 
    # TODO make this functional.
    # For now, assign label proportional to the distance from the origin.
    Y = []
    for x in X:
        prob_of_1 = 1 - np.linalg.norm(x) ** 10
        #norm = scipy.stats.multivariate_normal.pdf(x)
        #prob_of_1 = x
        label = 0
        if random.random() < prob_of_1:
            label = 1
        Y.append(label)
    return np.array(Y)


def write_dataset(X, Y, outfile):
    """Writes the dataset to the outfile."""
    with h5py.File(outfile, 'w') as f:
        f[DEFAULT_DATA_KEY] = X
        f[DEFAULT_LABEL_KEY] = Y


def plot_dataset(X, Y, outfile):
    """Plots the data and writes it to outfile."""
    if len(X[0]) != 2:
        raise ValueError('Cannot display non-2D data.')
    x0 = np.array([x[0] for x in X])
    x1 = np.array([x[1] for x in X])
    colors = ['blue' if y == 0 else 'red' for y in Y]
    fig, ax = plt.subplots()
    ax.scatter(x0, x1, c=colors)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.savefig(outfile)
    plt.show()


def get_args():
    """Returns the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=DEFAULT_NUM_EXAMPLES,
        help='The number of samples in the output dataset.')
    parser.add_argument('--m', type=int, default=DEFAULT_NUM_FEATURES,
        help='The number of features in the output dataset.')
    parser.add_argument('--outfile', type=str, default=DEFAULT_OUTFILE,
        help='The name of the output file to produce.')
    parser.add_argument('--plot_file', type=str, default=DEFAULT_PLOT_FILE,
        help='The name of the plot file to produce.')
    return parser.parse_args()


def main():
    """Runs the program."""
    args = get_args()
    dim = (args.n, args.m)
    X = get_random_points(dim)
    print('X is of shape: {0}'.format(X.shape))
    Y = get_labels(X)
    write_dataset(X, Y, args.outfile)
    print('Dataset written to: {0}'.format(args.outfile))
    print('Plotting dataset.')
    plot_dataset(X, Y, args.plot_file)


if __name__ == '__main__':
    main()


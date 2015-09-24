#!/usr/bin/env python
#-*- coding: utf-8 -*-
from __future__ import division
from os.path import splitext
import sys
import argparse
from hicexplorer import HiCMatrix as hm
from hicexplorer.utilities import enlarge_bins
from hicexplorer._version import __version__

from scipy import sparse
import numpy as np


def parsearguments(args=None):
    """
    get command line arguments
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Uses the graph clustering measure "conductance" to '
        'find minimum cuts that correspond to boundaries.')

    # define the arguments
    parser.add_argument('--matrix', '-m',
                        help='matrices to use.',
                        metavar='.npz fileformat',
                        required=True)

    parser.add_argument('--depth',
                        help='window length to be considered left and right '
                        'of the cut point in bp',
                        metavar='INT bp',
                        type=int,
                        )

    parser.add_argument('--threshold',
                        help='noise threshold',
                        type=float,
                        )

    parser.add_argument('--outFileName', '-o',
                        help='File name to save the values',
                        type=argparse.FileType('w'),
                        required=True)

    parser.add_argument('--version', action='version',
                        version='%(prog)s {}'.format(__version__))

    return parser.parse_args(args)


def get_min_indices(conductance):
    """returns the indices corresponding to the
    local minimum found on the given array.

    """
    conductance = np.array(conductance)
    # find nan conductance and replace by average
    for nan_idx in np.flatnonzero(np.isnan(conductance)):
        try:
            conductance[nan_idx] = np.mean([conductance[nan_idx-1],
                                            conductance[nan_idx+1]])
        except IndexError:
            conductance[nan_idx] = 0

    min_indices = np.array(
        [idx for idx in range(1, len(conductance) - 2)
         if conductance[idx-1] > conductance[idx] < conductance[idx+1]])

    if len(min_indices) == 0:
        exit("no local minimum found")

    return min_indices


def save_boundaries_and_domains(min_indices, conductance_array, filename,
                                threshold=None, window_length=3):

    assert len(min_indices) > 0, 'no min indices'
    chrom, start, end, conductance = zip(*conductance_array)
    conductance = np.array(conductance)
    # get difference between the local minimun found
    # and the neighbors average to detect if the minimum
    # represents a significant difference with respect
    # to the neighboring conductance

    # compare the local minimum with the highest of the 'window_length'
    # left and right neighbors
    """
    diff = {idx:max(
            (conductance[idx-window_length-1:idx+window_length] - 
             conductance[idx]))
            for idx in min_indices if idx > window_length and
            idx < len(conductance)}

    """
    diff = {idx:(max((conductance[idx-window_length-1:idx] - conductance[idx])),
                 max((conductance[idx:idx+window_length] - conductance[idx])))
            for idx in min_indices if idx > window_length and
            idx < len(conductance)}

    if threshold is None:
        threshold_range = np.arange(0.0, 0.01, 0.0005)
    else:
        threshold_range = [threshold]

    for threshold_value in threshold_range:
#        min_idx = np.sort([idx for idx in diff.keys() if diff[idx] > threshold_value])
        min_idx = np.sort([idx for idx in diff.keys()
                           if (diff[idx][0] > threshold_value
                           and diff[idx][1] > threshold_value)
                           or (diff[idx][0] > 3*threshold_value
                           and diff[idx][1] > 0.5*threshold_value)
                           or (diff[idx][0] > 0.5*threshold_value
                           and diff[idx][1] > 3*threshold_value) ])

        if len(min_idx) == 0:
            continue
        if len(threshold_range) > 1:
            file_name = splitext(filename)[0] + \
                "_{}_boundaries.bed".format(threshold_value)
            file_domains = splitext(filename)[0] + \
                "_{}_domains.bed".format(threshold_value)
        else:
            file_name = splitext(filename)[0] + \
                "_boundaries.bed".format(threshold_value)
            file_domains = splitext(filename)[0] + \
                "_domains.bed".format(threshold_value)

        fileh = open(file_name, 'w')
        filed = open(file_domains, 'w')
        prev_start = 0
        count = 0
        for idx in min_idx:
            boundary_center = end[idx] - int((end[idx] - start[idx])/2)
            # this happens at the borders of chromosomes
            if boundary_center < 0:
                continue
            fileh.write("{0}\t{1}\t{2}\tID_{3}\t{4}\t.\n".format(
                    chrom[idx],
                    boundary_center,
                    boundary_center + 1,
                    idx,
                    conductance[idx]))

            if count % 2 == 0:
                rgb = '51,160,44'
            else:
                rgb = '31,120,180'
            filed.write("{0}\t{1}\t{2}\tID_{3}\t{4}\t."
                        "\t{1}\t{2}\t{5}\n".format(chrom[idx],
                                                   prev_start,
                                                   boundary_center,
                                                   idx,
                                                   conductance[idx],
                                                   rgb))
            count += 1
            prev_start = boundary_center

        fileh.close()
        filed.close()


def get_cut_weight(matrix, cut, depth):
    """
    Get inter cluster edges sum.
    Computes the sum of the counts
    between the left and right regions of a cut

    >>> matrix = np.array([
    ... [ 0,  0,  0,  0,  0],
    ... [10,  0,  0,  0,  0],
    ... [ 5, 15,  0,  0,  0],
    ... [ 3,  5,  7,  0,  0],
    ... [ 0,  1,  3,  1,  0]])

    Test a cut at position 2, depth 2.
    The values in the matrix correspond
    to:
          [[ 5, 15],
           [ 3,  5]]
    >>> get_cut_weight(matrix, 2, 2)
    28

    For the next test the expected
    submatrix is [[10],
                  [5]]
    >>> get_cut_weight(matrix, 1, 2)
    15
    >>> get_cut_weight(matrix, 4, 2)
    4
    >>> get_cut_weight(matrix, 5, 2)
    0
    """
    # the range [start:i] should have running window
    # length elements (i is excluded from the range)
    start = max(0, cut - depth)
    # same for range [i+1:end] (i is excluded from the range)
    end = min(matrix.shape[0], cut + depth)

    # the idea is to evaluate the interactions
    # between the upstream neighbors with the
    # down stream neighbors. In other words
    # the inter-domain interactions
    return  matrix[cut:end, :][:, start:cut].sum()


def get_min_volume(matrix, cut, depth):
    """
    The volume is the weight of the edges
    from a region to all other.

    In this case what I compute is
    a submatrix that goes from
    cut - depth to cut + depth
    """
    start = max(0, cut - depth)
    # same for range [i+1:end] (i is excluded from the range)
    end = min(matrix.shape[0], cut + depth)

    left_region = matrix[start:end, :][:, start:cut].sum()
    right_region = matrix[cut:end, :][:, start:end].sum()

    return min(left_region, right_region)

def get_conductance(matrix, cut, depth):
    """
    The conductance is defined as the
    inter-domain edges / min(edges from left, edges from right)

    only computed for a submatrix corresponding
    to a running window of length 2*depth

    The matrix has to be lower or uppper to avoid
    double counting

    >>> matrix = np.array([
    ... [ 0,  0,  0,  0,  0],
    ... [10,  0,  0,  0,  0],
    ... [ 5, 15,  0,  0,  0],
    ... [ 3,  5,  7,  0,  0],
    ... [ 0,  1,  3,  1,  0]])

    The left intra counts are '10',
    The right intra counts are '7',
    The inter counts are:
          [[ 5, 15],
           [ 3,  5]], sum = 28

    >>> res = get_conductance(matrix, 2, 2)
    >>> res == 28.0 / 35
    True
    """
    start = max(0, cut - depth)
    # same for range [i+1:end] (i is excluded from the range)
    end = min(matrix.shape[0], cut + depth)

    inter_edges = get_cut_weight(matrix, cut, depth)
    edges_left = inter_edges + matrix[start:cut, :][:, start:cut].sum()
    edges_right = inter_edges + matrix[cut:end, :][:, cut:end].sum()

    return float(inter_edges) / min(edges_left, edges_right)
#    return float(inter_edges) / (sum([edges_left, edges_right]) - inter_edges)


def get_coverage(matrix, cut, depth):
    """
    The coverage is defined as the
    intra-domain edges / all edges

    I only computed for a small running window
    of length 2*depth

    The matrix has to be lower or uppper to avoid
    double counting
    """
    start = max(0, cut - depth)
    # same for range [i+1:end] (i is excluded from the range)
    end = min(matrix.shape[0], cut + depth)

    cut_weight = get_cut_weight(matrix, cut, depth)
    total_edges = matrix[start:end, :][:, start:end].sum()
    return cut_weight / total_edges


def smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len, 'd')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(), s, mode='same')
    return y[window_len:-window_len+1]


def main(args):
    hic_ma = hm.hiCMatrix(args.matrix)

#    hic_ma.keepOnlyTheseChr('chr3L')
#    sys.stderr.write("\nWARNING: using only chromosome 3L\n\n")
    # remove self counts
#    hic_ma.diagflat(value=0)
    sys.stderr.write('keeping diagonal values\n')
    hic_ma.matrix.data = np.log(hic_ma.matrix.data)
    sys.stderr.write('using log matrix values\n')

    hic_ma.maskBins(hic_ma.nan_bins)
    orig_intervals = hic_ma.cut_intervals
    new_intervals = enlarge_bins(hic_ma.cut_intervals)
    if new_intervals != orig_intervals:
        hic_ma.interval_trees, hic_ma.chrBinBoundaries = \
            hic_ma.intervalListToIntervalTree(new_intervals)

    if args.depth % hic_ma.getBinSize() != 0:
        sys.stderr.write('Warning. specified depth is not multiple of the '
                         'hi-c matrix bin size ({})\n'.format(
                hic_ma.getBinSize()))
    binsize = hic_ma.getBinSize()
    depth_in_bins = int(args.depth / binsize)
    sys.stderr.write('Using a window of {} bins\n'.format(depth_in_bins))
    if depth_in_bins <= 1:
        sys.stderr.write('window length too small\n')
        exit()
#    hic_ma.matrix, nan_bins = fill_gaps(hic_ma, depth_in_bins)
    # mask from matrix the remaining nan bins

    # work only with the lower matrix, and skip
    # direct contacts (diagonal k=-1)
    hic_ma.matrix = sparse.tril(hic_ma.matrix, k=0, format='csr')

    conductance_array = []

    for cut in range(1, hic_ma.matrix.shape[0]-1):
        conductance = get_conductance(hic_ma.matrix, cut, depth_in_bins)
#        conductance = get_coverage(hic_ma.matrix, cut, depth_in_bins)

        chrom, chr_start, chr_end, _ = hic_ma.cut_intervals[cut]

        # the evaluation of the conductance happens
        # at the position between bins, thus the
        # conductance is stored in bins that
        # span the neighboring bins. In other
        # words, the coductance is evaluated
        # at the position between let's say
        # bins number 14 and 15. Instead of
        # storing a score at the position in between
        # bin 14 and bin 15, a region of the size
        # of the bins, centered on the interface.
        if chr_start - int(binsize/2) > 0:
            chr_start -= int(binsize/2)
        else:
            chr_start = 0

        chr_end -= int(binsize/2)
        conductance_array.append((chrom, chr_start, chr_end, conductance))

    chrom, chr_start, chr_end, conductance = zip(*conductance_array)

    smooth_window_length = int(6000/binsize)

    conductance = smooth(np.array(conductance),
                         window_len=smooth_window_length)
    min_indices = get_min_indices(conductance)
    with open(splitext(args.outFileName.name)[0] + '_min_positions.bed', 'w') as f:
        for idx in min_indices:
            f.write("{}\t{}\t{}\t{}\n".format(chrom[idx], chr_start[idx],
                                              chr_end[idx], conductance[idx]))

    # window_length is measured in number of bins
    # but makes more sense to related to genomic distance
    window_length = int(15000/binsize)
    if window_length < 3:
        window_length = 4
    sys.stderr.write("Using window_length = {}\n".format(window_length))
    conductance_array = zip(chrom, chr_start, chr_end, conductance)
    # save boundary positions in bed format
    save_boundaries_and_domains(min_indices, conductance_array,
                                args.outFileName.name,
                                threshold=args.threshold,
                                window_length=window_length)

    for interval in conductance_array:
        args.outFileName.write("{}\t{}\t{}\t{}\n".format(*interval))
    args.outFileName.close()

if __name__ == "__main__":
    ARGS = parsearguments()
    main(ARGS)
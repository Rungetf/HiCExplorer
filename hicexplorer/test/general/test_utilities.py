import os
import pytest
import argparse
import numpy.testing as nt
import numpy as np
from hicmatrix import HiCMatrix as hm
from hicexplorer import utilities as utils


ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_data/utilities")


def test_writableFile(capsys):
    writeable_file = ROOT + '/writeable_file.txt'
    read_only_file = ROOT + '/readonly_file.txt'

    utils.writableFile(writeable_file)

    with pytest.raises(argparse.ArgumentTypeError):
        utils.writableFile(read_only_file)
        out, err = capsys.readouterr()
        assert out == "{} file can be opened for writing".format(read_only_file)
        assert err == argparse.ArgumentTypeError(out)


def test_remove_outliers():
    data = [i for i in range(100)]
    test_data = []
    new_data = utils.remove_outliers(data)
    assert new_data == test_data


def test_convertNansToZeros():
    matrix = np.array([[np.nan, np.nan, np.nan],
                        [0, 0, 0]])

    new = utils.convertNansToZeros(matrix.todense())

    test_matrix = np.array([[0, 0, 0],
                             [0, 0, 0]])

    nt.assert_equal(new, test_matrix)


def test_convertInfsToZeros():
    matrix = np.array([[np.inf, np.inf, np.inf],
                        [0, 0, 0]])

    new = utils.convertInfsToZeros(matrix)

    test_matrix = np.array([[0, 0, 0],
                             [0, 0, 0]])

    nt.assert_equal(new, test_matrix)


def test_convertInfsToZeros_ArrayFloat():
    matrix = np.array([[np.inf, np.inf, np.inf],
                        [0, 0, 0]])

    new = utils.convertInfsToZeros(matrix)

    test_matrix = np.array([[0, 0, 0],
                             [0, 0, 0]])

    nt.assert_equal(new, test_matrix)


def test_convertNansToOnes():
    matrix = np.array([[np.nan, np.nan, np.nan],
                        [0, 0, 0]])

    new = utils.convertNansToZeros(matrix)

    test_matrix = np.array([[1, 1, 1],
                             [0, 0, 0]])

    nt.assert_equal(new, test_matrix)


def test_myAverage():
    pass  # not used in any of the files.


def test_enlarge_bins():
    bin_intervals = [('chr1', 10, 50, 1), ('chr1', 50, 80, 2), ('chr2', 10, 60, 3), ('chr2', 70, 90, 4)]

    new = utils.enlarge_bins(bin_intervals)
    test = [('chr1', 0, 50, 1), ('chr1', 50, 80, 2), ('chr2', 0, 65, 3), ('chr2', 65, 90, 4)]

    nt.assert_equal(new, test)


def test_genomicRegion():
    strings = ['chr1:10-50', 'chr1:10,50', 'chr1:10:50']
    test = 'chr1:10:50'

    for string in strings:
        new = utils.genomicRegion(string)
        assert new == test


def test_getUserRegion():
    data = utils.getUserRegion({'chr1': 1000}, "chr1:10:10")

    new = utils.getUserRegion({'chr2': 1000}, "chr2:10:1001")
    test = ([('chr2', 1000)], 10, 1000, 990)
    nt.assert_equal(new, test)

    # Test chunk and regions size reduction to match tile size
    new = utils.getUserRegion({'chr2': 200000}, "chr2:10:123344:3")
    test = ([('chr2', 123344)], 9, 123345, 123336)
    nt.assert_equal(new, test)


def test_expected_interactions_in_distance():
    pass

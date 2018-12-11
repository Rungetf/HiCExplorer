import warnings
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=PendingDeprecationWarning)
import os.path
from tempfile import NamedTemporaryFile
from hicexplorer import hicConvertFormat
import pytest
from hicmatrix import HiCMatrix as hm
import numpy.testing as nt
import numpy as np

REMOVE_OUTPUT = True
# DIFF = 60

DELTA_DECIMAL = 0

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_data/hicConvertFormat")
original_matrix_h5 = ROOT + "/small_test_matrix.h5"
original_matrix_cool = ROOT + "/small_test_matrix.cool"
original_matrix_homer = ROOT + "/small_test_matrix.homer"


@pytest.mark.parametrize("matrices", [original_matrix_h5, original_matrix_cool])  # required
@pytest.mark.parametrize("outputFormat", ['cool', 'h5', 'homer', 'ginteractions', 'mcool'])
@pytest.mark.parametrize("resolutions", ['', '--resolutions 5000', '--resolutions 5000 10000', '--resolutions 5000 10000 20000'])
def test_trivial_run(
    matrices,
    outputFormat,
    resolutions,
):
    """
        Test for all commandline arguments.
        Options for cool input format are testet seperately.
    """
    from pathlib import Path
    # get suffix of input matrix without the dot
    inputFormat = Path(matrices).suffix[1:]
    # create file corresponding to output format
    outFileName = NamedTemporaryFile(suffix=".{}".format(outputFormat), delete=True)

    args = "--matrices {} --outFileName {} --inputFormat {} --outputFormat {} {}".format(
        matrices,
        outFileName.name,
        inputFormat,
        outputFormat,
        resolutions,
    ).split()

    hicConvertFormat.main(args)


@pytest.mark.parametrize("matrices", [original_matrix_h5, original_matrix_cool])  # required
@pytest.mark.parametrize("outputFormat", ['h5', 'cool', 'homer', 'ginteractions', 'mcool'])
@pytest.mark.parametrize("resolutions", ['', '--resolutions 5000', '--resolutions 5000 10000', '--resolutions 5000 10000 20000'])
def test_trivial_functionality(
    matrices,
    outputFormat,
    resolutions,
):
    """
        Test of functionality of all formats.
    """
    from pathlib import Path
    # get suffix of input matrix without the dot
    inputFormat = Path(matrices).suffix[1:]
    # create file corresponding to output format
    outFileName = NamedTemporaryFile(suffix=".{}".format(outputFormat), delete=False)
    outFileName.close()

    args = "--matrices {} --outFileName {} --inputFormat {} --outputFormat {} {}".format(
        matrices,
        outFileName.name,
        inputFormat,
        outputFormat,
        resolutions,
    ).split()

    hicConvertFormat.main(args)

    test = hm.hiCMatrix(matrices + "::/resolutions/5000")
    test_2 = hm.hiCMatrix(matrices + "::/resolutions/10000")
    test_3 = hm.hiCMatrix(matrices + "::/resolutions/20000")

    new = hm.hiCMatrix(outFileName.name + '::/resolutions/5000')
    new_2 = hm.hiCMatrix(outFileName.name + '::/resolutions/10000')
    new_3 = hm.hiCMatrix(outFileName.name + '::/resolutions/20000')


    nt.assert_array_almost_equal(test.matrix.data, new.matrix.data, decimal=DELTA_DECIMAL)
    nt.assert_array_almost_equal(test_2.matrix.data, new_2.matrix.data, decimal=DELTA_DECIMAL)
    nt.assert_array_almost_equal(test_3.matrix.data, new_3.matrix.data, decimal=DELTA_DECIMAL)


    nt.assert_equal(len(new.cut_intervals), len(test.cut_intervals))
    nt.assert_equal(len(new_2.cut_intervals), len(test_2.cut_intervals))
    nt.assert_equal(len(new_3.cut_intervals), len(test_3.cut_intervals))


    cut_interval_new_ = []
    cut_interval_test_ = []
    for x in new.cut_intervals:
        cut_interval_new_.append(x[:3])
    for x in test.cut_intervals:
        cut_interval_test_.append(x[:3])

    nt.assert_equal(cut_interval_new_, cut_interval_test_)

    cut_interval_new_ = []
    cut_interval_test_ = []
    for x in new_2.cut_intervals:
        cut_interval_new_.append(x[:3])
    for x in test_2.cut_intervals:
        cut_interval_test_.append(x[:3])

    nt.assert_equal(cut_interval_new_, cut_interval_test_)

    cut_interval_new_ = []
    cut_interval_test_ = []
    for x in new_3.cut_intervals:
        cut_interval_new_.append(x[:3])
    for x in test_3.cut_intervals:
        cut_interval_test_.append(x[:3])

    nt.assert_equal(cut_interval_new_, cut_interval_test_)

    os.unlink(outFileName.name)

import logging
import numpy as np
from numpy.core.records import fromarrays
from ..utils import process_vector

logger = logging.getLogger("bact2")


def unpack_and_validate_data(
    packed_data,
    n_valid_items=None,
    indices=None,
    n_elements=None,
    skip_unset_second_half=True,
):
    """Split the packed data up to a matrix of bpm waveforms

    Args :
        packed_data : the data as received by the BPM

    Returns :
        mat : a matrix with the  different signals as row vectors

    See :func:`packed_data_to_named_array` for convenient procecssing of packed
    data to named array
    """

    if n_elements is None:
        # BESSY II standard
        n_elements = 2048

    assert n_valid_items is not None
    # Test for digital twind
    shape = packed_data.shape
    (ndim,) = shape
    if ndim != n_elements:
        txt = f"Expected a vector with shape {n_elements} but got an array with shape {shape}"
        raise AssertionError(txt)

    assert ndim == n_elements
    if skip_unset_second_half:
        # remove this functionality from here
        shape = packed_data.shape
        try:
            mat1 = process_vector.unpack_vector_to_matrix(packed_data, n_vecs=2)
        except:
            logger.error(f"{__name__}: Failed to unpack packed data with shape {shape}")
            raise

        with_data = mat1[0, :]
        unused = mat1[1, :]

        process_vector.check_unset_elements(
            unused, n_valid_rows=n_valid_items, unset_elements_value=0
        )
        del unused

    else:
        with_data = packed_data

    mat = process_vector.unpack_vector_to_matrix(with_data, n_vecs=8)
    if indices is not None:
        shape = mat.shape
        ishape = indices.shape
        try:
            mat = np.take(mat, indices, axis=1)
        except:
            logger.error(
                f"{__name__}: Failed to process mat with shape {shape} using indices with shape {ishape}"
            )
            logger.error(f"indices: {indices}")
            raise
    return mat

    # Excess elements are not checked in the different subvectors as
    # these contain 'random data'


def data_to_named_array(mat):
    """preprocessed packed data to named array

    See :func:`packed_data_to_named_array` for convenient procecssing of packed
    data to named array

    Args:
        mat: a matrix with the appropriate number of elements

    """
    names = ["x_pos_raw", "y_pos_raw", "intensity_z", "intensity_s", "stat"]
    names += ["gain_raw", "x_rms_raw", "y_rms_raw"]

    # At least the first dimension
    rows = [mat[row] for row in range(mat.shape[0])]
    data = fromarrays(rows, names=names)

    return data


def packed_data_to_named_array(
    packed_data,
    n_valid_items=None,
    indices=None,
    n_elements=None,
    skip_unset_second_half=True,
):
    """check packed bpm data and split it to a named array"""
    mat = unpack_and_validate_data(
        packed_data,
        n_valid_items=n_valid_items,
        indices=indices,
        n_elements=n_elements,
        skip_unset_second_half=skip_unset_second_half,
    )
    t_array = data_to_named_array(mat)
    return t_array


#: default bit gain for bpms. Scales raw readings to mm scale
#: These can be still wrong by factors, but are then in the appropriate decade
bit_gain_default = 2**15 / 10.0


def raw_to_scaled_data_channel(values, gain, offset, bit_gain=None):
    """BPM raw to physics coordinates

    BPM data are first scaled from raw data to mm.
    Then the offset is subtracted.
    """

    if bit_gain is None:
        t_gain = gain
    else:
        t_gain = gain * bit_gain

    del gain, bit_gain

    scaled = values / t_gain
    r = scaled - offset
    return r


def raw_to_scaled_data(a_array, bpm_parameters, bit_gain=None):
    """Scale the channel data for the bpms

    Args:
       a_array: a record array containing the raw data of the bpm's
       bpm_parameters: a record array containing the rescale values
       bit_gain: conversion factor from bit to gain parameters
    """

    conv = raw_to_scaled_data_channel

    x_scale = bpm_parameters["x_scale"]
    x_offset = bpm_parameters["x_offset"]
    x_pos = conv(a_array["x_pos_raw"], x_scale, x_offset, bit_gain=bit_gain)
    x_rms = conv(a_array["x_rms_raw"], x_scale, 0.0, bit_gain=bit_gain)

    del x_scale, x_offset

    y_scale = bpm_parameters["y_scale"]
    y_offset = bpm_parameters["y_offset"]
    y_pos = conv(a_array["y_pos_raw"], y_scale, y_offset, bit_gain=bit_gain)
    y_rms = conv(a_array["y_rms_raw"], y_scale, 0.0, bit_gain=bit_gain)

    del y_scale, y_offset

    t_names = ["x_pos", "y_pos", "x_rms", "y_rms"]
    res = fromarrays([x_pos, y_pos, x_rms, y_rms], names=t_names)
    return res


def packed_data_to_scaled(
    packed_data, *, bpm_parameters, n_valid_items=128, bit_gain=bit_gain_default
):
    """scaled bpm data from packed data

    Returns: raw, scaled

    The returned arrays are:
       * raw: the packed data split up to its different parts.
              These different parts are stored in a record
              array

       * scaled: the raw data processed using the information in bpm_parameters
    """
    indices = bpm_parameters["idx"] - 1
    raw = packed_data_to_named_array(
        packed_data, n_valid_items=n_valid_items, indices=indices
    )
    scaled = raw_to_scaled_data(raw, bpm_parameters=bpm_parameters, bit_gain=bit_gain)

    return raw, scaled

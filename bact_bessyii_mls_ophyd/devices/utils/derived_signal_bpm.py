"""Linear signal transformation as done for BPM's

Code originally developed within bact2
"""
from .derived_signal import DerivedSignalLinear
from ..process.bpm_packed_data import (
    packed_data_to_named_array,
    raw_to_scaled_data_channel,
)
from ophyd import Component as Cpt, Device, Kind, Signal
from ophyd.status import AndStatus, DeviceStatus
import numpy as np
import logging

logger = logging.getLogger("bact")


class DerivedSignalLinearBPM(DerivedSignalLinear):
    """BPM raw data to signal

    The inverse is used for calculating the bpm offset
    in mm from the raw data.

    use_offset can be set to zero. This is used for
    recalculating rms values
    """

    def __init__(self, *args, use_offset=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_offset = bool(use_offset)

    def forward(self, values):
        raise NotImplementedError("Can not make a bpm a steerer")

    def inverse(self, values):
        """BPM raw to physics coordinates

        BPM data are first scaled from raw data to mm.
        Then the offset is subtracted.
        """

        gain = self._gain.get()
        bit_gain = self._bit_gain.get()

        if self.use_offset:
            offset = self._offset.get()
        else:
            offset = 0

        conv = raw_to_scaled_data_channel
        r = conv(values, gain, offset, bit_gain=bit_gain)
        return r


class BPMChannelScale(DerivedSignalLinearBPM):
    """Values required for deriving BPM reading from raw signals

    Derived signals require to be informed which signal name to use
    as channel source. For the BPM's this recalculation is
    implemented using a linear transformation. Please note that the
    *inverse* transform is used to transform the read signals to
    millimeters.

    The following three signals are used:
        * gain
        * bit_gain
        * offset

    Compared to a
    Reduce repetition of typing for

    """

    def __init__(self, *args, **kwargs):

        for sig_name in ["gain", "bit_gain", "offset"]:
            kwargs.setdefault("parent_{}_attr".format(sig_name), sig_name)

        super().__init__(*args, **kwargs)


class BPMChannel(Device):
    """A channel (or coordinate) of the bpm

    The beam position monitor reading is split in the coordinates
        * x
        * y

    This is made, as each channel requires the following signals:
        * pos:      the actual position
        * rms:      the rms of the actual position
        * pos_raw:  raw reading of the position
        * rms_raw:  rms of the raw reading
        * gain:     a vector for rescaling the device from
        * bit_gain: a rough scale from mm to bit

    Warning:
        Let :class:`BPMWavefrom` use it
        It is the users responsibility to set the gains correctly!
    """

    _default_config_attrs = ("gain", "offset", "scale")

    #: Relative beam offset as measured by the beam position monitors
    pos_raw = Cpt(Signal, name="pos_raw")
    #: and its rms value
    rms_raw = Cpt(Signal, name="rms_raw")

    #: gains for the channels
    gain = Cpt(Signal, name="gain", value=1.0, kind=Kind.config)

    #: offset of the BPM from the ideal orbit
    offset = Cpt(Signal, name="offset", value=0.0, kind=Kind.config)

    #: scale bits to mm
    bit_gain = Cpt(Signal, name="bit_gain", value=2**15 / 10, kind=Kind.config)

    #: processed data: already in mm
    pos = Cpt(BPMChannelScale, parent_attr="pos_raw", name="pos")
    rms = Cpt(BPMChannelScale, parent_attr="rms_raw", name="rms", use_offset=False)

    def trigger(self):
        raise NotImplementedError("Use BPMWaveform instead")


class BPMWaveform(Device):
    """Measurement data for the beam position monitors

    Todo:
        Reference to the coordinate system
        Clarify status values
        Clarify why data are given for "non existant monitors"
    """

    #: Number of valid beam position monitors
    n_valid_bpms = None

    # number of elements to expect
    n_elements = None

    # is there a second unused half on the data
    skip_unset_second_half = None

    #: All data for x
    x = Cpt(BPMChannel, "x")
    #: All data for y
    y = Cpt(BPMChannel, "y")

    #: Data not sorted into the different channels
    intensity_z = Cpt(Signal, name="z")
    intensity_s = Cpt(Signal, name="s")
    status = Cpt(Signal, name="status")

    #: gains as found in the packed data. The gains for recalculating
    #: the values are found in the BPMChannels
    gain_raw = Cpt(Signal, name="gain")

    ds = Cpt(Signal, name="ds", value=np.nan, #kind=Kind.config
    )
    names = Cpt(Signal, name="names", value=[], kind=Kind.config)
    indices = Cpt(Signal, name="indices", value=[], kind=Kind.config)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.n_elements is not None
        assert self.skip_unset_second_half is not None

        # Check its there and an int
        assert self.n_valid_bpms is not None
        self.n_valid_bpms = int(self.n_valid_bpms)
        assert self.n_valid_bpms > 0

    ##     self.setConfigData()
    ##
    ## def setConfigData(self):
    ##     rec = create_bpm_config()
    ##     self.names.put(rec['name'])
    ##     self.ds.put(rec['ds'])
    ##     idx = rec['idx']
    ##     self.indices.put(idx - 1)
    ##     self.x.gain.put(rec['x_scale'])
    ##     self.y.gain.put(rec['y_scale'])
    ##     self.x.offset.put(rec['x_offset'])
    ##     self.y.offset.put(rec['y_offset'])

    def storeDataInWaveforms(self, array):
        """Store row vectors to the appropriate signals

        Args:
            mat : a matrix of vectors containing the appropriate input data
        Todo:
            Analyse the status values !
            To be removed
        """

        self.x.pos_raw.put(array["x_pos_raw"])
        self.y.pos_raw.put(array["y_pos_raw"])
        self.x.rms_raw.put(array["x_rms_raw"])
        self.y.rms_raw.put(array["y_rms_raw"])

        self.intensity_z.put(array["intensity_s"])
        self.intensity_s.put(array["intensity_z"])
        self.status.put(array["stat"])
        self.gain_raw.put(array["gain_raw"])

    def checkAndStorePackedData(self, packed_data):

        indices = self.indices.get()
        if len(indices) == 0:
            indices = None
        else:
            self.log.debug(f"len indices {len(indices)}")
            
        array = packed_data_to_named_array(
            packed_data,
            n_valid_items=self.n_valid_bpms,
            n_elements=self.n_elements,
            skip_unset_second_half=self.skip_unset_second_half,
            indices=indices,
        )
        li = len(indices)
        if array.shape[0] != li:
            self.log.warning(f"Expected array shape of [{li}, .] but got {array.shape}")
            
        return self.storeDataInWaveforms(array)

    def trigger(self):
        status_processed = DeviceStatus(self, timeout=5)

        def check_data(*args, **kws):
            """Check that the received data match the expected ones

            Could also be done during read status. I like to do it here
            as I see it as part of ensuring that good data were
            received
            """

            nonlocal status_processed

            data = self.packed_data.get()
            self.checkAndStorePackedData(data)
            # Not acceptable for ophyd versions to come
            # status_processed.success = True
            status_processed._finished()

        status = super().trigger()
        status.add_callback(check_data)

        and_s = AndStatus(status, status_processed)
        return and_s

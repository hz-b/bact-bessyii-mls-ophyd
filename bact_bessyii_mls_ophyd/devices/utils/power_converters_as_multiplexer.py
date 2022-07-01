from ophyd import Signal, Device, Component as Cpt, Kind

from ophyd import PseudoPositioner, PseudoSingle
from ophyd.pseudopos import pseudo_position_argument, real_position_argument
from .power_converter import ResettingPowerConverter

class ScaledPowerConverter(PseudoPositioner):
    """A configurable steerer

    It allows changing its offset and slope a configuration variables
    """

    # The pseudo positioner axes:
    p = Cpt(PseudoSingle, limits=(-1, 1))

    # The real (or physical) positioners:
    # I guess the empty space is significant
    r = Cpt(ResettingPowerConverter, "", name="r")

    offset = Cpt(Signal, name="offset", value=0.0, kind=Kind.config)
    slope = Cpt(Signal, name="slope", value=1.0, kind=Kind.config)

    @pseudo_position_argument
    def forward(self, pseudo_pos):
        """Run a forward (pseudo -> real) calculation"""

        offset = self.offset.get()
        slope = self.slope.get()

        v = pseudo_pos.p
        r = v * slope + offset

        return self.RealPosition(r=r)

    @real_position_argument
    def inverse(self, real_pos):
        """Run an inverse (real -> pseudo) calculation"""

        offset = self.offset.get()
        slope = self.slope.get()

        offset = float(offset)
        slope = float(slope)
            
        r = real_pos.r
        try:
            tmp = r - offset
        except:
            self.log.error(f"Value r was {r}")
            raise
        
        v = tmp / slope

        return self.PseudoPosition(p=r)


_selected_default = "none selected"


class SelectedPowerConverter(Device):
    """Select a steerer and mangle its data ..."""

    selected = Cpt(Signal, name="selected", value=_selected_default)

    def getSelectedName(self):
        """ """
        name = self.selected.get()
        if name != _selected_default:
            return name

        names = self.parent.power_converter_names.get()
        default_name = names[0]
        self.log.warning(f"Setting default steerer to {default_name}")
        stat = self.selected.set(default_name)
        stat.wait(1)
        name = default_name
        return name

    def _setSelectedPowerConverterByName(self, name):
        """
        Todo:
            check if the set value should moved from the last selected to this one
        """
        try:
            steerer = getattr(self.parent.power_converters, name)
            self._sel = steerer
        except Exception as e:
            fmt = "{}._setSteererByName failed to set steerer {} reason {}"
            self.log.error(fmt.format(self.name, name, e))
            raise
        self.log.debug("Set Steerer: selected steerer {}".format(steerer))
        t_name = str(name)
        stat = self.selected.set(t_name)
        return stat

    def getSelectedPowerConverter(self):
        sel = None
        try:
            sel = self._sel
        except AttributeError:
            pass

        if sel is not None:
            return sel

        name = self.getSelectedName()
        stat = self._setSelectedPowerConverterByName(name)
        stat.wait(1)

        sel = self._sel
        return sel

    def set(self, val):
        """Just to make it easier to run it with bluesky

        Args:
            name : steerer name
        """
        sel = self.getSelectedPowerConverter()
        status = sel.set(val)
        return status

    # def trigger_all_components_update(self):
    #    status = None
    #    for name in self.steerers.component_names:
    #        cpt = getattr(self.steerers, name)
    #        sig_rdbk = cpt.readback
    #        r = trigger_on_update(sig_rdbk)
    #        if status is None:
    #            status = r
    #        else:
    #            status = AndStatus(status, r)
    #    return status

    def _renameKeys(self, d):
        sel = self.getSelectedPowerConverter()
        sel_name = sel.name
        prefix = self.name
        replace_prefix = prefix  # + '_' + 'sc_selected'
        fmt = 'Renaming prefix from "%s" to "%s"'
        self.log.debug(fmt, prefix, replace_prefix)

        def rename_key(key):
            tmp, r = key.split(sel_name)
            assert tmp == ""
            new_key = replace_prefix + r
            return new_key

        nd = {rename_key(key): item for key, item in d.items()}
        return nd

    def _applyMethodToSelected(self, method_name):
        sel = self.getSelectedPowerConverter()
        method = getattr(sel, method_name)
        r = method()
        return r

    def _applyMethodAndRename(self, method_name):
        d = self._applyMethodToSelected(method_name)
        d = self._renameKeys(d)
        return d

    def describe_configuration(self):
        d = super().describe_configuration()
        d2 = self._applyMethodAndRename("describe_configuration")
        d.update(d2)
        return d2

    def read_configuration(self):
        d = super().read_configuration()
        d2 = self._applyMethodAndRename("read_configuration")
        d.update(d2)
        return d2

    def describe(self):
        d = super().describe()
        d2 = self._applyMethodAndRename("describe")
        d.update(d2)
        return d

    def trigger(self):
        sel = self.getSelectedPowerConverter()
        status = sel.trigger()
        fmt = "%s.trigger: %s "
        self.log.debug(fmt, self.name, status)
        return status

    def read(self):
        d = super().read()
        d2 = self._applyMethodAndRename("read")
        d.update(d2)
        return d

    def stop(self, success=False):
        sel = self.getSelectedPowerConverter()
        sel.stop(success=success)


class MultiplexerSetMixin:
    def set(self, name):
        """Just to make it easier to run it with bluesky

        Args:
            name : steerer name
        """
        self.log.info("Selecting power converter {}".format(name))
        status = self.sel._setSelectedPowerConverterByName(name)
        return status


__all__ = ["ScaledPowerConverter", "SelectedPowerConverter", "MultiplexerSetMixin"]

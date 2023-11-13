import super_state_machine.machines
import enum


class MuxerState(super_state_machine.machines.StateMachine):
    """State of the muxer"""

    class States(enum.Enum):
        """ """

        #: e.g. at start up. Here the object has to find out in which state
        #: the muxer is
        UNDEFINED = "undefined"
        #: muxer off: muxer power converter not connected to any magnet
        OFF = "off"
        #: a magnet is selected
        SELECTED = "selected"
        #: switched off but expecting to switch over to a magnet
        #: that's typically the transition from one magnet to
        #: an other one
        #:
        #: we start at a selected magnet.
        #: set it to off_for_switch
        #: and as soon as this signal arrives we switch over to
        #: intermediate
        OFF_FOR_SWITCH = "off_for_switch"
        #: now should be off. the muxer is now expected
        #: to transit to the
        INTERMEDIATE = "intermediate"
        #: Failed: something not as expected
        FAILED = "failed"

    class Meta:
        initial_state = "undefined"
        transitions = {
            #: default at start up
            "undefined": ["undefined", "off", "selected"],
            #: when off muxer switches directly to selected
            "off": ["off", "selected"],
            #: "selected" : either
            "selected": ["off", "off_for_switch", "selected"],
            #: from selected state, expecting to switch to off for switch
            #: then to intermediate
            "off_for_switch" : ["intermediate"],
            #: intermediate expecting to switch to a selected magnet
            "intermediate": ["selected"],
            #: if failed: set back to undefiend
            "failed": ["undefined"],
        }

        # call failed if an error is detected ...
        named_transitions = [("failed", "failed")]

"""
File created by Samuel Bakker.
Contains the Slot-class used in simulation.py.
"""
class Slot:
    """
    Class for a slot in the Simulation.

    Attributes
    ----------
    startTime: float
        start time of the slot (in hours)
    appTime: float
        appointment time of the slot, dependent on type and rule (in hours)
    slotType: int
        type of slot (0=none, 1=elective, 2=urgent within normal working hours, 3=urgent in overtime)
    patientType: int
        (0=none, 1=elective, 2=urgent)
    """

    startTime: float
    appTime: float
    slotType: int
    patientType: int

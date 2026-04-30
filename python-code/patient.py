"""
File created by Samuel Bakker.
Contains the Patient-class used in simulation.py.
"""

class Patient:
    """Patient class

    Attributes
    ----------
    nr: int
        id
    patientType: int
        (0=none, 1=elective, 2=urgent)
    scanType: int
        elective: (0=none), urgent: (0=brain, 1=lumbar, 2=cervival, 3=abdomen, 4=others)
    callWeek: int
        week of arrival (elective: call, urgent: actual arrival)
    callDay: int
        day of arrival (elective: call, urgent: actual arrival)
    callTime: float
        time of arrival (elective: call, urgent: actual arrival) (in hours)
    scanWeek: int
        week of appointment
    scanDay: int
        day of appointment
    slotNr: int
        slot number of appointment
    appTime: float
        time of appointment (elective: according to rule, urgent: slot start time) (in hours)
    tardiness: float
        in hours
    isNoShow: bool
    scanTime: float
        actual start time of the scan
    duration: float
        actual duration of the scan

    Methods
    ----------
        _type_: _description_
    """
    nr: int  # id
    patientType: int  # (0=none, 1=elective, 2=urgent)
    scanType: int  # elective: (0=none), urgent: (0=brain, 1=lumbar, 2=cervival, 3=abdomen, 4=others)
    callWeek: int  # week of arrival (elective: call, urgent: actual arrival)
    callDay: int  # day of arrival (elective: call, urgent: actual arrival)
    callTime: float  # time of arrival (elective: call, urgent: actual arrival) (in hours)
    scanWeek: int  # week of appointment
    scanDay: int  # day of appointment
    slotNr: int  # slot number of appointment
    appTime: float  # time of appointment (elective: according to rule, urgent: slot start time) (in hours)
    tardiness: float  # (in hours)
    isNoShow: bool
    scanTime: float  # actual start time of the scan
    duration: float  # actual duration of the scan

    def __init__(self, nr: int, patientType: int, scanType: int, callWeek: int, callDay: int, callTime: int, tardiness: float, isNoShow: bool, duration: float) -> None:
        self.nr = nr
        self.patientType = patientType
        self.scanType = scanType
        self.callWeek = callWeek
        self.callDay = callDay
        self.callTime = callTime
        self.tardiness = tardiness
        self.isNoShow = isNoShow
        self.duration = duration
        # unplanned patients (all patients start like this)
        self.scanWeek = -1
        self.scanDay = -1
        self.slotNr = -1
        self.appTime = -1
        self.scanTime = -1.0

    def getAppWT(self) -> float:
        """
        Calculate the waiting time for the appointment of this specific patient.
        Depends on time of arrival, amount of weeks waited etc.
        Exits the entire program when the patient is an emergency patient.
        Returns:
            float: Waiting time for appointment
        """
        if (self.slotNr == -1):
            print(f"CAN NOT CALCULATE APPOINMTENT WAITING TIME OF PATIENT {self.nr}")
            exit(1)
        return float(((self.scanWeek - self.callWeek) * 7 + self.scanDay - self.callDay) * 24 + self.appTime - self.callTime)

    def getScanWT(self) -> float:
        """
        Calculate the waiting time for the scan for this specific patient.
        Depends on scheduled time, time of arrival

        Returns:
            float: Waiting time for scan
        """
        if (self.scanTime == 0):
            print(f"CAN NOT CALCULATE SCAN WAITING TIME OF PATIENT {self.nr}")
            exit(1)
        wt: float = 0
        if (self.patientType == 1):
            # elective patients
            wt = self.scanTime - (self.appTime + self.tardiness)
        else:
            wt = self.scanTime - self.callTime

        return float(max(0, wt))

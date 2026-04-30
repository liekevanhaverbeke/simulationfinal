"""
File created by Samuel Bakker.
Contains the Simulation-class used in simulation.py.

This is the file you should run.
Due to the "if __name__ == "__main__":"-statement.
"""
import re
import random
from functools import cmp_to_key

from helper import Exponential_distribution, Normal_distribution, Bernouilli_distribution
from slot import Slot
from patient import Patient
class Simulation:
    """
    Simulation instance

    Class Attributes
    ----------------
    inputFileName: str
    D: int
        number of days per week (NOTE: Sunday not included! so do NOT use to calculate appointment waiting time)
    amountOTSlotsPerDay: int
        number of overtime slots per day
    S: int
        number of slots per day
    slotLength: float
        duration of a slot (in hours)
    lamdaElective: float
    meanTardiness: float
    stdevTardiness: float
    probNoShow: float
    meanElectiveDuration: float
    stdevElectiveDuration: float
    lambdaUrgent: tuple[float]
    probUrgentType: tuple[float]
    cumulativeProbUrgentType: tuple[float]
    meanUrgentDuration: tuple[float]
    stdevUrgentDuration: tuple[float]
    weightEl: float
        objective weight elective appointment wait time
    weightUr: float
        objective weight urgent scan wait time

    Attributes
    ----------
    W: int
        Number of weeks
    R: int
        Numer of replications
    d: int
    s: int
    w: int
    r: int
    rule: int
        appointment scheduling rule
    weekSchedule: list[list[Slot]]
        list of cyclic slot schedule
    patients: list[Patient]
        list of patients
    movingAvgElectiveAppWT: list[float]
        moving average of elective appointment waiting times
    movingAvgElectiveScanWT: list[float]
        moving average elective scan waiting time
    movingAvgUrgentScanWT: list[float]
        moving average urgent scan waiting time
    movingAvgOT: list[float]
        moving average overtime
    avgElectiveAppWT: float
        average elective appointment waiting time
    avgElectiveScanWT: float
        average elective scan waiting time
    avgUrgentScanWt: float
        average urgent scan waiting time
    avgOT: float
        average overtime
    numberOfElectivePatientsPlanned: int
    numberOfUrgentPatientsPlanned: int
    """
    inputFileName: str
    D: int = 6
    amountOTSlotsPerDay: int = 10
    S: int = 32 + amountOTSlotsPerDay
    slotLength: float = float(15 / 60)
    lambdaElective: float = 28.345
    meanTardiness: float = 0
    stdevTardiness: float = 2.5
    probNoShow: float = 0.02
    meanElectiveDuration: float = 15
    stdevElectiveDuration: float = 3
    lambdaUrgent: tuple[float] = (2.5, 1.25)
    probUrgentType: tuple[float] = (0.7, 0.1, 0.1, 0.05, 0.05)
    cumulativeProbUrgentType: tuple[float] = (0.7, 0.8, 0.9, 0.95, 1.0)
    meanUrgentDuration: tuple[float] = (15, 17.5, 22.5, 30, 30)
    stdevUrgentDuration: tuple[float] = (2.5, 1, 2.5, 1, 4.5)
    weightEl: float = 1.0 / 168.0
    weightUr: float = 1.0 / 9.0

    W: int
    R: int
    d: int
    s: int
    w: int
    r: int
    rule: int
    weekSchedule: list[list[Slot]]

    # var within one simulation
    patients: list[Patient]
    movingAvgElectiveAppWT: list[float]
    movingAvgElectiveScanWT: list[float]
    movingAvgUrgentScanWT: list[float]
    movingAvgOT: list[float]
    avgElectiveAppWT: float
    avgElectiveScanWT: float
    avgUrgentScanWt: float
    avgOT: float
    numberOfElectivePatientsPlanned: int
    numberOfUrgentPatientsPlanned: int

    def __init__(self, filename: str, W: int, R: int, rule: int) -> None:
        """
        Constructor. Used to instantiate a simulation.

        Args:
            filename (str): Name of the inputfile to read in, containing a schedule
            W (int): Number of weeks to simulate (aka run length)
            R (int): Number of replications
            rule (int): The appointment scheduling rule to apply
        """
        self.patients = list()
        self.inputFileName = filename
        self.W = W
        self.R = R
        self.rule = rule

        self.avgElectiveAppWT = 0
        self.avgElectiveScanWT = 0
        self.avgUrgentScanWt = 0
        self.avgOT = 0
        self.numberOfElectivePatientsPlanned = 0
        self.numberOfUrgentPatientsPlanned = 0

        # initialize weekSchedule. 6 days (excl. sunday)
        self.weekSchedule = []
        for d in range(self.D):
            self.weekSchedule.append([Slot() for s in range(self.S)])

        self.movingAvgElectiveAppWT = list()
        self.movingAvgElectiveScanWT = list()
        self.movingAvgUrgentScanWT = list()
        self.movingAvgOT = list()

    def generatePatients(self) -> None:
        """
        Create new patients and add them to the list of patients for the current simulation object.
        """
        arrivalTimeNext = 0.0
        counter = 0
        for w in range(self.W):
            for d in range(self.D):
                # Start off by generating elective patients
                if (d < self.D - 1):  # not on sunday
                    arrivalTimeNext = 8 + Exponential_distribution(self.lambdaElective) * (17 - 8)
                    while (arrivalTimeNext < 17):
                        tardiness = Normal_distribution(self.meanTardiness, self.stdevTardiness) / 60
                        noShow = Bernouilli_distribution(self.probNoShow)
                        duration = Normal_distribution(self.meanElectiveDuration, self.stdevElectiveDuration) / 60
                        # create a patient with all the calculated data and add it to the list for simulation:
                        # they arrive in the current week (outer loop: w) at the current day (inner loop: d)
                        # all patients have a time of arrival (arrivalTimeNext), can be late or early (tardiness),
                        # they also have a probability of not showing up at all (noShow), and a given duration for their procedure
                        self.patients.append(Patient(counter, 1, 0, w, d, arrivalTimeNext, tardiness, noShow, duration))
                        counter += 1
                        arrivalTimeNext += Exponential_distribution(self.lambdaElective) * (17 - 8)

                lmbd = self.lambdaUrgent[0]
                endTime = 17
                # radiology dept is only open half a day on thursday and saturday
                # change the values if needed:
                if ((d == 3) or (d == 5)):
                    lmbd = self.lambdaUrgent[1]
                    endTime = 12
                arrivalTimeNext = 8 + Exponential_distribution(lmbd) * (endTime - 8)
                while (arrivalTimeNext < endTime):
                    noShow = 0  # Urgent patients always show up, would be silly otherwise
                    tardiness = 0  # Urgent patients are not planned and therefore cannot be late
                    scanType = self.getRandomScanType()
                    duration = Normal_distribution(self.meanUrgentDuration[scanType], self.stdevUrgentDuration[scanType]) / 60
                    self.patients.append(Patient(counter, 2, scanType, w, d, arrivalTimeNext, tardiness, noShow, duration))
                    counter += 1
                    arrivalTimeNext += Exponential_distribution(lmbd) * (endTime - 8)

    def getRandomScanType(self) -> int:
        """
        Generate a random scanType for a patient.
        Used for Urgent patients, since the type of scan needed is unknown a priori

        Returns:
            int: integer corresponding to the type of scan
        """
        r = random.random()
        for idx, prob in enumerate(self.cumulativeProbUrgentType):
            if r < prob:
                return idx

    def getNextSlotNrFromTime(self, day: int, patientType: int, time: float) -> int:
        """
        Get the next available timeSlot based on the current time in the simulation

        Args:
            day (int): day of the simulation
            patientType (int): type (urgent or not)
            time (float): time of day

        Returns:
            int: next slotnumber
        """
        for s in range(self.S):
            if ((self.weekSchedule[day][s].appTime > time) and (patientType == self.weekSchedule[day][s].patientType)):
                return s
        print(f"NO SLOT EXISTS DURING TIME {time} \n")
        exit(0)

    @staticmethod
    def sortPatientsOnAppTime(patient1: Patient, patient2: Patient) -> None:
        """Sorting function

        Args:
            patient1 (Patient): left item
            patient2 (Patient): right item

        Returns:
            None: sorts list in place
        """
        # unscheduled patients:
        if ((patient1.scanWeek == -1) and (patient2.scanWeek == -1)):
            if (patient1.callWeek < patient2.callWeek):
                return -1
            if (patient1.callWeek > patient2.callWeek):
                return 1
            # if same week, look at days:
            if (patient1.callDay < patient2.callDay):
                return -1
            if (patient1.callDay > patient2.callDay):
                return 1
            # if same day, look at time:
            if (patient1.callTime < patient2.callTime):
                return -1
            if (patient1.callTime > patient2.callTime):
                return 1
            # if the arrival time is also the same, then urgent patients (type 2) get preference
            if (patient1.patientType == 2):
                return -1
            if (patient2.patientType == 2):
                return 1
            return 0
        if (patient1.scanWeek == -1):
            # patient1 is not scheduled yet, move backwards
            return 1
        if (patient2.scanWeek == -1):
            # patient2 is not scheduled yet, move backwards
            return -1
        # scheduled patients:
        if (patient1.scanWeek < patient2.scanWeek):
            return -1
        if (patient1.scanWeek > patient2.scanWeek):
            return 1
        if (patient1.scanDay < patient2.scanDay):
            return -1
        if (patient1.scanDay > patient2.scanDay):
            return 1
        if (patient1.appTime < patient2.appTime):
            return -1
        if (patient1.appTime > patient2.appTime):
            return 1
        # if arrival time is the same, look at urgency:
        if (patient1.patientType == 2):
            return -1
        if (patient2.patientType == 2):
            return 1
        # all the above is probably also largely summarized in the following two statements:
        if (patient1.nr < patient2.nr):
            return -1
        if (patient1.nr > patient2.nr):
            return 1
        return 0

    @staticmethod
    def sortPatients(patient1: Patient, patient2: Patient) -> int:
        # sort based on weeks
        if (patient1.callWeek < patient2.callWeek):
            return -1
        if (patient1.callWeek > patient2.callWeek):
            return 1
        # same week, look at days:
        if (patient1.callDay < patient2.callDay):
            return -1
        if (patient1.callDay > patient2.callDay):
            return 1
        # if same day, look at time of arrival:
        if (patient1.callTime < patient2.callTime):
            return -1
        if (patient1.callTime > patient2.callTime):
            return 1
        # if arrival time is the same, then urgent patients are prioritized
        if (patient1.patientType == 2):
            return -1
        if (patient2.patientType == 2):
            return 1
        return 0

    def schedulePatients(self) -> None:
        """
        Used in the sorting function.
        """
        self.patients = sorted(self.patients, key=cmp_to_key(Simulation.sortPatients))
        # now we look for the first available slot of every patient type
        week = [0, 0]  # week of next available slot
        day = [0, 0]  # day of next available slot
        slot = [0, 0]  # slotNr of next available slot

        # assumption: every day has atleast one slot of eacht patient type (so also day 0)
        # elective:
        for s in range(self.S):
            if (self.weekSchedule[0][s].patientType == 1):
                day[0] = 0
                slot[0] = s
                break
        # urgent
        for s in range(self.S):
            if (self.weekSchedule[0][s].patientType == 2):
                day[1] = 0
                slot[1] = s
                break

        previousWeek = 0
        numberOfElectivePerWeek = 0
        numberOfElective = 0
        for patient in self.patients:
            # i is used to index the w and d list.
            # this is because we only need to plan within a certain horizon (namely W amount of weeks)
            i = patient.patientType - 1
            if (week[i] < self.W):
                # look if week and day need to be updated:
                if (patient.callWeek > week[i]):
                    week[i] = patient.callWeek
                    day[i] = 0
                    # assume there is at least one slot of each patient type per day => this line will find first slot of this type
                    slot[i] = self.getNextSlotNrFromTime(day[i], patient.patientType, 0)
                elif ((patient.callWeek == week[i]) and (patient.callDay > day[i])):
                    day[i] = patient.callDay
                    slot[i] = self.getNextSlotNrFromTime(day[i], patient.patientType, 0)
                # get slot
                if ((patient.callWeek == week[i]) and (patient.callDay == day[i]) and (patient.callTime >= self.weekSchedule[day[i]][slot[i]].appTime)):
                    # as every day has all types of patienttype slots, we can look for the last slot of a certain type as follows:
                    for s in range(self.S - 1, -1, -1):
                        if (self.weekSchedule[day[i]][s].patientType == patient.patientType):
                            # this if-statement will always fire, because of the fact that all types are present in a day
                            slotNr = s
                            break
                    if ((patient.patientType == 2) or (patient.callTime < self.weekSchedule[day[i]][slotNr].appTime)):
                        slot[i] = self.getNextSlotNrFromTime(day[i], patient.patientType, patient.callTime)
                    else:
                        if (day[i] < self.D - 1):
                            day[i] = day[i] + 1
                        else:
                            day[i]
                            week[i] += 1
                        if (week[i] < self.W):
                            slot[i] = self.getNextSlotNrFromTime(day[i], patient.patientType, 0)
                # schedule patient
                patient.scanWeek = week[i]
                patient.scanDay = day[i]
                patient.slotNr = slot[i]
                patient.appTime = self.weekSchedule[day[i]][slot[i]].appTime

                if (patient.patientType == 1):
                    if (previousWeek < week[i]):
                        self.movingAvgElectiveAppWT[previousWeek] /= numberOfElectivePerWeek
                        numberOfElectivePerWeek = 0
                        previousWeek = week[i]
                    wt = patient.getAppWT()
                    self.movingAvgElectiveAppWT[week[i]] += wt
                    numberOfElectivePerWeek += 1
                    self.avgElectiveAppWT += wt
                    numberOfElective += 1

                # set next slot of the current patient type
                found = False
                startD = day[i]
                startS = slot[i] + 1
                for w in range(week[i], self.W):
                    for d in range(startD, self.D):
                        for s in range(startS, self.S):
                            if (self.weekSchedule[d][s].patientType == patient.patientType):
                                week[i] = w
                                day[i] = d
                                slot[i] = s
                                found = True
                                break
                        if (found):
                            break
                        startS = 0  # next day
                    if (found):
                        break
                    startD = 0  # next week
                if (not found):
                    week[i] = self.W
        self.movingAvgElectiveAppWT[self.W - 1] /= numberOfElectivePerWeek
        self.avgElectiveAppWT /= numberOfElective

    def runOneSimulation(self) -> None:
        """
        Run one simulation.
        1. Generatie new patients
        2. Schedule patients
        3. Sort the patients
        4. Then for every patient get the waiting time
        """
        self.generatePatients()
        self.schedulePatients()
        self.patients = sorted(self.patients, key=cmp_to_key(Simulation.sortPatientsOnAppTime))
        prevWeek = 0
        prevDay = -1
        numberOfPatientsWeek = [0, 0]
        numberOfPatients = [0, 0]
        prevScanEndTime = 0
        prevIsNoShow = False

        tard = 0  # check tardiness
        for patient in self.patients:
            if (patient.scanWeek == -1):
                # stop when a patient is not scheduled
                break

            arrivalTime = patient.appTime + patient.tardiness
            # if a patient shows up, then this ofcourse has an impact on our waiting times, so do the calc:
            if (not patient.isNoShow):
                if ((patient.scanWeek != prevWeek) or (patient.scanDay != prevDay)):
                    patient.scanTime = arrivalTime
                else:
                    if (prevIsNoShow):
                        patient.scanTime = max(self.weekSchedule[patient.scanDay][patient.slotNr].startTime, max(prevScanEndTime, arrivalTime))
                    else:
                        patient.scanTime = max(prevScanEndTime, arrivalTime)
                wt = patient.getScanWT()
                if (patient.patientType == 1):
                    # if non urgent
                    self.movingAvgElectiveScanWT[patient.scanWeek] += wt
                    self.avgElectiveScanWT += wt
                else:
                    self.movingAvgUrgentScanWT[patient.scanWeek] += wt
                    self.avgUrgentScanWt += wt
                # update count of patienttype this week
                numberOfPatientsWeek[patient.patientType - 1] += 1
                numberOfPatients[patient.patientType - 1] += 1
            # for overtime, day 3 and 5 are halfdays:
            if ((prevDay > -1) and (prevDay != patient.scanDay)):
                if ((prevDay == 3) or (prevDay == 5)):
                    self.movingAvgOT[prevWeek] += max(0, prevScanEndTime - 13)
                    self.avgOT += max(0.0, prevScanEndTime - 13)
                else:
                    self.movingAvgOT[prevWeek] += max(0, prevScanEndTime - 17)
                    self.avgOT += max(0.0, prevScanEndTime - 17)

            if (prevWeek != patient.scanWeek):
                self.movingAvgElectiveScanWT[prevWeek] /= numberOfPatientsWeek[0]
                self.movingAvgUrgentScanWT[prevWeek] /= numberOfPatientsWeek[1]
                self.movingAvgOT[prevWeek] /= self.D
                numberOfPatientsWeek[0] = 0
                numberOfPatientsWeek[1] = 0
            if (patient.isNoShow):
                # for a noshow, the prevscantime does not change bc the last time someone was scanned doesnot changed
                prevIsNoShow = True
                if ((patient.scanWeek != prevWeek) or (patient.scanDay != prevDay)):
                    prevScanEndTime = self.weekSchedule[patient.scanDay][patient.slotNr].startTime
            else:
                prevScanEndTime = patient.scanTime + patient.duration
                prevIsNoShow = False
            prevWeek = patient.scanWeek
            prevDay = patient.scanDay
            tard += patient.tardiness
        # moving averages of lastweek
        self.movingAvgElectiveScanWT[self.W - 1] /= numberOfPatientsWeek[0]
        self.movingAvgUrgentScanWT[self.W - 1] /= numberOfPatientsWeek[1]
        self.movingAvgOT[self.W - 1] /= self.D
        # calc objective values
        self.avgElectiveScanWT /= numberOfPatients[0]
        self.avgUrgentScanWt /= numberOfPatients[1]
        self.avgOT /= (self.D * self.W)

    def setWeekSchedule(self) -> None:
        """
        This method sets a cyclic slot schedule based on a given input file and applied rules.

        @Students, if you choose to use Python, make sure to implement the required rules here.
        """
        # read file:
        # NOTE we assume utf-8-sig, as many students will probably be working with MS
        with open(self.inputFileName, 'r', encoding='utf-8-sig') as r:
            slotTypes = list(map(lambda x: re.findall('[0-9]', x), r.readlines()))
            assert len(slotTypes) == 32, "Error: there should be 32 slots (lines) in the file"
            for slotIdx, weekSlot in enumerate(slotTypes):
                assert len(weekSlot) == self.D, f"Error: there should be {self.D} days in the file (columns)"
                for slotDayIdx, inputInteger in enumerate(weekSlot):
                    self.weekSchedule[slotDayIdx][slotIdx].slotType = int(inputInteger)
                    self.weekSchedule[slotDayIdx][slotIdx].patientType = int(inputInteger)

        # set type of overtime slots (3 is urgent OT)
        for d in range(self.D):
            for s in range(32, self.S):
                self.weekSchedule[d][s].slotType = 3
                self.weekSchedule[d][s].patientType = 2

        for d in range(self.D):
            time = 8
            for s in range(self.S):
                # start time slot
                self.weekSchedule[d][s].startTime = time
                # appointment time slot
                if (self.weekSchedule[d][s].slotType != 1):
                    self.weekSchedule[d][s].appTime = time
                else:
                    if (self.rule == 1):
                        # FIFO
                        self.weekSchedule[d][s].appTime = time
                    elif (self.rule == 2):
                        # DONE: Bailey-Welch rule
                        K_rule = 2
                        elective_count = sum(1 for prev_s in range(s) if self.weekSchedule[d][prev_s].slotType == 1)
                        first_elective_time = next(self.weekSchedule[d][ps].startTime for ps in range(s + 1) if self.weekSchedule[d][ps].slotType == 1)
                        if elective_count < K_rule:
                            self.weekSchedule[d][s].appTime = first_elective_time
                        else:
                            self.weekSchedule[d][s].appTime = time - self.slotLength
                    elif (self.rule == 3):
                        # DONE: Blocking rule
                        B_rule = 2
                        elective_count = sum(1 for prev_s in range(s) if self.weekSchedule[d][prev_s].slotType == 1)
                        block_start_rank = (elective_count // B_rule) * B_rule
                        elective_slots = [ps for ps in range(s + 1) if self.weekSchedule[d][ps].slotType == 1]
                        self.weekSchedule[d][s].appTime = self.weekSchedule[d][elective_slots[block_start_rank]].startTime
                    elif (self.rule == 4):
                        # DONE: Benchmark rule
                        self.weekSchedule[d][s].appTime = time - 0.5 * self.stdevElectiveDuration / 60
                time += self.slotLength
                if (time == 12):
                    # Lunchbreak, so skip ahead
                    time = 13

    def resetSystem(self) -> None:
        """
        Resets all variables that are related to 1 replication.
        """
        self.patients = list()
        self.avgElectiveAppWT = 0.0
        self.avgElectiveScanWT = 0.0
        self.avgUrgentScanWt = 0.0
        self.avgOT = 0.0
        self.numberOfElectivePatientsPlanned = 0
        self.numberOfUrgentPatientsPlanned = 0

        # test this;
        self.movingAvgElectiveAppWT = []
        self.movingAvgElectiveScanWT = []
        self.movingAvgUrgentScanWT = []
        self.movingAvgOT = []
        for w in range(self.W):
            self.movingAvgElectiveAppWT.append(0.0)
            self.movingAvgElectiveScanWT.append(0.0)
            self.movingAvgUrgentScanWT.append(0.0)
            self.movingAvgOT.append(0.0)

    def runSimulations(self) -> None:
        """
        Function that runs all the simulations.

        @Students: Create the functionality to write the output to a file in here.
        This is not too difficult. Google helps a lot!
        """
        electiveAppWT: float = 0
        electiveScanWT: float = 0
        urgentScanWT: float = 0
        OT: float = 0
        OV: float = 0
        self.setWeekSchedule()
        print("r \t elAppWT \t elScanWT \t urScanWT \t OT \t OV \n")
        for r in range(self.R):
            self.resetSystem()
            random.seed(r)
            self.runOneSimulation()
            electiveAppWT += self.avgElectiveAppWT
            electiveScanWT += self.avgElectiveScanWT
            urgentScanWT += self.avgUrgentScanWt
            OT += self.avgOT
            OV += self.avgElectiveAppWT * self.weightEl + self.avgUrgentScanWt * self.weightUr
            print(f"{r} \t {self.avgElectiveAppWT:.2f} \t\t {self.avgElectiveScanWT:.5f} \t {self.avgUrgentScanWt:.2f} \t\t {self.avgOT:.2f} \t {self.avgElectiveAppWT * self.weightEl + self.avgUrgentScanWt * self.weightUr:.2f}")
        electiveAppWT /= self.R
        electiveScanWT /= self.R
        urgentScanWT /= self.R
        OT /= self.R
        OV /= self.R
        print("--------------------------------------------------------------------------------")
        print(f"AVG: \t {electiveAppWT:.2f} \t\t {electiveScanWT:.5f} \t {urgentScanWT:.2f} \t\t {OT:.2f} \t {OV:.2f} \n")

if __name__ == "__main__":
    sim = Simulation("../input-S1-14.txt", 100, 100, 1)
    sim.runSimulations()

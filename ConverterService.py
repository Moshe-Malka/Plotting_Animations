
import csv
from io import StringIO
import pandas as pd

# EXAM_FIELDS = ["ExamTime", "Time", "BirthDate", "PatientId", "Protocol", "ScreenWidth", "ScreenHight", "PixelsInCM"]
# TEST_FIELDS = ["Index", "TestTime", "TestType", "PictureName", "TellerType", "Duration", "LeftEye", "RightEye", "Animated", "X", "Y", "Width", "Height"]
TEST_FIELDS = ["Index", "TestType", "TellerType", "LeftEye", "RightEye", "X", "Y", "Width", "Height"]
# TRACKER_FIELDS = ["T", "lx", "ly", "lv", "rx", "ry", "rv", "lpd", "rpd", "lpx", "lpy", "lpz", "rpx", "rpy", "rpz", "fx", "fy", "fv"]
TRACKER_FIELDS = ["lx", "ly", "lv", "rx", "ry", "rv", "lpd", "rpd", "lpz", "rpz"]
DEFAULT_ID = "99999"

def convert(json_data):
    try:
        with StringIO() as buf:
            # writer = csv.DictWriter(buf, EXAM_FIELDS + TEST_FIELDS + TRACKER_FIELDS)
            writer = csv.DictWriter(buf, TEST_FIELDS + TRACKER_FIELDS)
            writer.writeheader()
            rec = {}
            # rec.update({f : json_data.get(f) for f in EXAM_FIELDS})
            # if not rec.get("PatientId"): rec["PatientId"] = DEFAULT_ID      # Handle a case when 'PatientId' is empty/None.
            tests = json_data.get("Tests", [])      # Grab all of the tests
            for test in tests:          # itirate over them
                rec.update({f: test.get(f) for f in TEST_FIELDS})            # for each test, update the record holder with the specific test general data. 
                trackers = test.get("EyeTracker", [])           # Grab all EyeTracker data for this specific test.
                for tracker in trackers:
                    rec.update({f: tracker.get(f) for f in TRACKER_FIELDS})           # Update our record holder with the specific EyeTracker data
                    writer.writerow(rec)                # Write each EyeTracker data as a row.
            return pd.read_csv(StringIO(buf.getvalue()))
    except Exception as e:
        print(f"Failed to load and/or convert raw data from json!\n{e}")
        return pd.DataFrame()
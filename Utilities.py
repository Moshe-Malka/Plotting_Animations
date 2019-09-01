
import csv
from io import StringIO
import pandas as pd

def convert(data):
    """ converting json data to csv format and than to pandas DataFrame. """
    exam_fields = ["ScreenWidth", "ScreenHight"]
    test_fields = ["Index", "TestType", "TellerType", "LeftEye", "RightEye", "X", "Y", "Width", "Height"]
    tracker_fields = ["lx", "ly", "lv", "rx", "ry", "rv", "lpd", "rpd", "lpz", "rpz"]
    try:
        with StringIO() as buf:
            writer = csv.DictWriter(buf, exam_fields + test_fields + tracker_fields)
            writer.writeheader()
            rec = {}
            rec.update({f : data.get(f) for f in exam_fields})
            tests = data.get("Tests", [])      # Grab all of the tests
            for test in tests:          # itirate over them
                rec.update({f: test.get(f) for f in test_fields})            # for each test, update the record holder with the specific test general data. 
                trackers = test.get("EyeTracker", [])           # Grab all EyeTracker data for this specific test.
                for tracker in trackers:
                    rec.update({f: tracker.get(f) for f in tracker_fields})           # Update our record holder with the specific EyeTracker data
                    writer.writerow(rec)                # Write each EyeTracker data as a row.
            return pd.read_csv(StringIO(buf.getvalue()))
    except Exception as e:
        print(f"Failed to load and/or convert raw data from json!\n{e}")
        return pd.DataFrame()

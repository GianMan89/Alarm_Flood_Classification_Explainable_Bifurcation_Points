import pandas as pd
import numpy as np
import os

import warnings

warnings.filterwarnings("ignore")


def load_data(data_folder: str):
    # Initialize empty lists for data and labels
    data = []
    labels = []
    # alarm subsequence classes
    ak_list = ["AK1", "AK2", "AK3", "AK4", "AK5"]
    # iterate over alarm subsequence classes
    for c, l in zip(ak_list, list(range(len(ak_list)))):
        # Get the list of files in the data folder
        data_files = os.listdir(data_folder + c)[:]
        # Load data and labels
        for file in data_files[:]:
            # Load data
            file_path = os.path.join(data_folder + c, file)
            # load the data using a pandas dataframe and convert to a numpy array
            data.append(pd.read_csv(file_path, header=None).values.transpose())
            labels.append(l)
    # Convert data and labels to numpy arrays
    X = np.array(data)
    y = np.array(labels)

    # Set the size of the test
    size_test = 1000

    # Randomly draw size_test_class samples for the test set
    test_idx = np.random.choice(X.shape[0], size_test, replace=False)
    # Use the remaining samples for the train set
    train_idx = np.setdiff1d(np.arange(X.shape[0]), test_idx)
 
    return (X, y, test_idx, train_idx)

# Get coverage for each class
def get_coverage_by_class(prediction_sets, y_test, y):
    coverage = []
    for i in np.unique(y):
        if i in y_test:
            coverage.append(np.mean(prediction_sets[y_test.flatten() == i, i]))
        else:
            coverage.append(np.nan)
    return coverage

# Get average set size for each class
def get_average_set_size(prediction_sets, y_test, y):
    average_set_size = []
    for i in np.unique(y):
        if i in y_test:
            average_set_size.append(
                np.mean(np.sum(prediction_sets[y_test.flatten() == i], axis=1))
            )
        else:
            average_set_size.append(np.nan)
    return average_set_size

class Alarm:
    """
    Alarm class
    """

    def __init__(self, type, start, end):
        self.sampling = 0.0166666666666667  # 1min
        self.start = start * self.sampling
        self.end = end * self.sampling
        self.type = type
        self.len = end - start + self.sampling

    def calc_len(self):
        self.len = self.end - self.start + self.sampling

    def __gt__(self, other):
        if self.start > other.start:
            return True
        else:
            return False

    def __ge__(self, other):
        if self.start >= other.start:
            return True
        else:
            return False

    def __lt__(self, other):
        if self.start < other.start:
            return True
        else:
            return False

    def __le__(self, other):
        if self.start <= other.start:
            return True
        else:
            return False

    def __eq__(self, other):
        if id(self) == id(other):
            return True
        else:
            return False


def convert_alarms(alarm_data: np.ndarray) -> list:
    """
    Convert alarm data to list of Alarm class
    """
    converted_alarm_data = []
    for j in range(alarm_data.shape[0]):
        alarm_list = []
        for i in range(alarm_data.shape[1]):
            idxs = np.where(alarm_data[j, i, :] == 1)[0]
            diffs = np.diff(idxs)
            if diffs.size == 0:
                continue
            elif max(diffs) == 1:
                alarm_list.append(Alarm(i, idxs[0], idxs[-1]))
            else:
                alarm_ends = idxs[np.where(diffs != 1)[0]]
                alarm_ends = np.array(
                    list(set(np.append(alarm_ends, [idxs[-1]])))
                )
                alarm_starts = idxs[np.where(diffs != 1)[0] + 1]
                alarm_starts = np.array(
                    list(set(np.append(alarm_starts, [idxs[0]])))
                )
                for start, end in zip(alarm_starts, alarm_ends):
                    alarm_list.append(Alarm(i, start, end))
        converted_alarm_data.append(sorted(alarm_list))
    return converted_alarm_data


from pathlib import Path
import re
import csv
import pandas as pd
import cv2

class ExtractionModule:
    def __init__(self):
        self._data = None

    def extract_data(self, log_file_path: Path, video_file_path: Path):
        cap = cv2.VideoCapture(video_file_path)
        fpms = cap.get(cv2.CAP_PROP_FPS) / 1000
        mspf = 1/fpms
        cap.release()

        data = {'x': [], 'y': [], 'z': [], 't': []}
        time = 0
        next_entry_is_fucked = False
        for line in open(log_file_path, 'r'):
            if line.startswith("[ DEBUG ] Face bounding box: ["):
                pattern = r"\[ DEBUG \] Face bounding box: \[(\d+) x \d+ from \(\d+, \d+\)\]"
                match = re.search(pattern, line)
                if match:
                    first_number = match.group(1)
                    if int(first_number) < 150:
                        next_entry_is_fucked = True
                        
            if line.startswith("[ DEBUG ] Gaze vector (x, y, z):"):
                if next_entry_is_fucked:
                    next_entry_is_fucked = False
                else:
                    numbers = re.findall(r'\[([-0-9e., ]+)\]', line)
                    if numbers:
                        list_of_numbers = [float(number.strip()) for number in numbers[0].split(',')]
                        data['x'].append(list_of_numbers[0])
                        data['y'].append(list_of_numbers[1])
                        data['z'].append(list_of_numbers[2])
                        data['t'].append(time)
                        time += mspf
                    else:
                        print("WARNING: some numbers didn't match the regular expression")
        self._data = pd.DataFrame(data)
        # print("< 200", maxi_less)
        # print(" > 200", min_gr)
        # print("count inferences", count_inferences)

    def print_summary(self):
        if type(self._data) != type(None):
            print(f'{len(self._data)} rows')
            print('First 5 rows:')
            print(self._data.head())
        else:
            print(self._data)

    def save_data_as_csv(self, csv_file_path: Path):
        self._data.to_csv(csv_file_path, index=False)

e = ExtractionModule()
# log_file_path = 'D:\ML\EyeC\PostProcessing\output\log_file_1'
# video_file_path = 'D:\ML\EyeC\PostProcessing\data\Demos\Demos\Day1\demo1_person1.MOD'
# csv_file_path1 = 'D:\ML\EyeC\PostProcessing\output\csv_file1.csv'
# e.extract_data(log_file_path, video_file_path)
# e.print_summary()
# e.save_data_as_csv(csv_file_path1)

# log_file_path = 'D:\ML\EyeC\PostProcessing\output\log_file_2'
# video_file_path = 'D:\ML\EyeC\PostProcessing\data\Demos\Demos\Day1\demo1_person2.MOD'
# csv_file_path1 = 'D:\ML\EyeC\PostProcessing\output\csv_file2.csv'
# e.extract_data(log_file_path, video_file_path)
# e.print_summary()
# e.save_data_as_csv(csv_file_path1)

# log_file_path = 'D:\ML\EyeC\PostProcessing\output\log_file_day_2_person1'
# video_file_path = 'D:\ML\EyeC\PostProcessing\data\Demos\Demos\Day2\demo2_person1.MOD'
# csv_file_path1 = 'D:\ML\EyeC\PostProcessing\output\csv_file_day2_1.csv'
# e.extract_data(log_file_path, video_file_path)
# e.print_summary()
# e.save_data_as_csv(csv_file_path1)

# log_file_path = 'D:\ML\EyeC\PostProcessing\output\log_file_day_2_person2'
# video_file_path = 'D:\ML\EyeC\PostProcessing\data\Demos\Demos\Day2\demo2_person2.MOD'
# csv_file_path1 = 'D:\ML\EyeC\PostProcessing\output\csv_file_day2_2.csv'
# e.extract_data(log_file_path, video_file_path)
# e.print_summary()
# e.save_data_as_csv(csv_file_path1)

log_file_path = 'D:\ML\EyeC\PostProcessing\output\log_file_cut_person1'
video_file_path = 'D:\ML\EyeC\PostProcessing\data\Demos\Demos\Cuts\demo1_person1_cut.mp4'
csv_file_path1 = 'D:\ML\EyeC\PostProcessing\output\csv_file_cut_person1.csv'
e.extract_data(log_file_path, video_file_path)
e.print_summary()
e.save_data_as_csv(csv_file_path1)

log_file_path = 'D:\ML\EyeC\PostProcessing\output\log_file_cut_person2'
video_file_path = 'D:\ML\EyeC\PostProcessing\data\Demos\Demos\Cuts\demo1_person2_cut.mp4'
csv_file_path1 = 'D:\ML\EyeC\PostProcessing\output\csv_file_cut_person2.csv'
e.extract_data(log_file_path, video_file_path)
e.print_summary()
e.save_data_as_csv(csv_file_path1)
from pathlib import Path
import re
import csv
import pandas as pd
import cv2
import numpy as np

class SynchronizationModule:
    def sync_files(self, sample_rate_ms: int, csv_file_path1: Path, csv_file_path2: Path, output_csv_file_path: Path):
        input_df1 = pd.read_csv(csv_file_path1)
        input_df2 = pd.read_csv(csv_file_path2)
        
        v1 = input_df1.to_numpy()
        v2 = input_df2.to_numpy()

        index_in_v1 = 0
        index_in_v2 = 0

        end_time = min(v1[-1][-1], v2[-1][-1])
        current_time = 0
        synced_v = []
        while current_time < end_time:
            while index_in_v1 + 1 < len(v1) and v1[index_in_v1 + 1][-1] <= current_time:
                index_in_v1 += 1

            while index_in_v2 + 1 < len(v2) and v2[index_in_v2 + 1][-1] <= current_time:
                index_in_v2 += 1

            entry = [current_time]
            entry.extend(v1[index_in_v1][:-1])
            entry.extend(v2[index_in_v2][:-1])
            
            synced_v.append(entry)
            # print(f"{current_time} | {v1[index_in_v1]} | {v2[index_in_v2]}")
            current_time += sample_rate_ms
        
        synced_df = pd.DataFrame(synced_v, columns = ['t', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2'])
        # print(len(synced_df))
        # print(synced_df.tail())

        synced_df.to_csv(output_csv_file_path, index=False)


        
s = SynchronizationModule()
# path1 = Path("D:\ML\EyeC\PostProcessing\output\csv_file1.csv")
# path2 = Path("D:\ML\EyeC\PostProcessing\output\csv_file2.csv")
# output_path = Path("D:\ML\EyeC\PostProcessing\output\synced_csv_file.csv")
# s.sync_files(30, path1, path2, output_path)

# path1 = Path("D:\ML\EyeC\PostProcessing\output\csv_file_day2_1.csv")
# path2 = Path("D:\ML\EyeC\PostProcessing\output\csv_file_day2_2.csv")
# output_path = Path("D:\ML\EyeC\PostProcessing\output\synced_csv_file_day2.csv")
# s.sync_files(30, path1, path2, output_path)

path1 = Path("D:\ML\EyeC\PostProcessing\output\csv_file_cut_person1.csv")
path2 = Path("D:\ML\EyeC\PostProcessing\output\csv_file_cut_person2.csv")
output_path = Path("D:\ML\EyeC\PostProcessing\output\synced_csv_file_cut.csv")
s.sync_files(30, path1, path2, output_path)
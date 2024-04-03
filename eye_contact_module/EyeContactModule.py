from pathlib import Path
import re
import csv
import pandas as pd
import cv2
from EyeContactClassifierModel import EyeContactClassifierModel
import json
import numpy as np
from typing import Dict, Tuple
import utils


class EyeContactModule:
    def __init__(self):
        pass


    def detect_eye_contact(self, config_file_path: Path, synced_csv_file_path: Path, output_csv_file_path: Path):        
        config = None
        with open(config_file_path, 'r') as json_file:
            config = json.load(json_file)

        config_aux = None
        with open("D:\ML\EyeC\PostProcessing\config\config_cut.json", 'r') as json_file:
            config_aux = json.load(json_file)

        if config is None:
            print("error: cannot parse config file")
            return

        synced_df = pd.read_csv(synced_csv_file_path)
        aux_df = pd.read_csv("D:\ML\EyeC\PostProcessing\output\synced_csv_file_cut.csv")
        data = synced_df.to_numpy()
        data_aux = aux_df.to_numpy()
        labeled_data, _ = self._split_data(data, config)
        _, unlabeled_data = self._split_data(data_aux, config_aux)
        
        model = EyeContactClassifierModel()
        model.fit(labeled_data)
        model.evaluate_on_labeled_data()
        # results = model.predict(np.concatenate((labeled_data[:, :-1], unlabeled_data)))
        results = model.predict(unlabeled_data)

        print(results)

        results_df = pd.DataFrame(results, columns=['t', 'class'])
        results_df['class'] = results_df['class'].replace({1: 'eye contact', 2: 'no eye contact', 3: 'eyes not detected'})

        results_df.to_csv(output_csv_file_path, index=False)


    def _split_data(self, data: np.ndarray, config: Dict) -> Tuple[np.ndarray, np.ndarray]:
        labeled_data = None
        unlabeled_data = None

        for key in config['phases'].keys():
            t_start = utils.time_string_to_ms(config['phases'][key]['start_time'])
            t_end =  utils.time_string_to_ms(config['phases'][key]['end_time'])

            filtered_data = self._filter_by_time(data, t_start, t_end)

            if key == 'conversation_phase':
                unlabeled_data = filtered_data
            else:
                labels_shape = len(filtered_data), 1
                if key == 'eye_contact_phase':
                    labels = np.full(labels_shape, 1) # 1 - eye contact class
                elif key == 'no_eye_contact_phase':
                    labels = np.full(labels_shape, 2) # 2 - no eye contact class

                new_labeled_data = np.concatenate((filtered_data, labels), axis=1)
                if labeled_data is None:
                    labeled_data = new_labeled_data
                else:
                    labeled_data = np.concatenate((labeled_data, new_labeled_data))
        
        return labeled_data, unlabeled_data
    

    def _filter_by_time(self, data: np.ndarray, period_start: float, period_end: float) -> np.ndarray:
        selected_indexes = np.logical_and(data[:, 0] >= period_start, data[:, 0] < period_end)
        return data[selected_indexes]


e = EyeContactModule()
# config_path = "D:\ML\EyeC\PostProcessing\config\config.json"
# input_path = "D:\ML\EyeC\PostProcessing\output\synced_csv_file.csv"
# output_path = "D:\ML\EyeC\PostProcessing\output\eye_contact_detection.csv"
# e.detect_eye_contact(config_path, input_path, output_path)

# config_path = "D:\ML\EyeC\PostProcessing\config\config_day2.json"
# input_path = "D:\ML\EyeC\PostProcessing\output\synced_csv_file_day2.csv"
# output_path = "D:\ML\EyeC\PostProcessing\output\eye_contact_detection_day2.csv"
# e.detect_eye_contact(config_path, input_path, output_path)


config_path = "D:\ML\EyeC\PostProcessing\config\config.json"
input_path = "D:\ML\EyeC\PostProcessing\output\synced_csv_file.csv"
output_path = "D:\ML\EyeC\PostProcessing\output\eye_contact_detection_cut.csv"
e.detect_eye_contact(config_path, input_path, output_path)
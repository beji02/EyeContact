import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from typing import Union, Tuple

class EyeContactClassifierModel:
    def __init__(self):
        pass
        self._model = KNeighborsClassifier(n_neighbors=10)
        self._labeled_data = None

    def fit(self, data: np.ndarray):
        self._labeled_data = data
        filtered_data = self._filter_undetected_eyes(data)
        X = filtered_data[:, 1:7]
        y = filtered_data[:, -1]
        self._model.fit(X, y)

    def predict(self, data: np.ndarray):
        X = data[:, 1:7]
        print(data[:, 0])
        results = self._model.predict(X)
        selected_indexes = self._get_indexes_of_undetected_eyes(data)
        results[selected_indexes] = 3 # 3 - undetected eyes
        results = np.expand_dims(results, axis=-1)
        results = np.concatenate((data[:, 0:1], results), axis = 1)

        return results

    def _get_indexes_of_undetected_eyes(self, data: np.ndarray) -> np.ndarray:
        selected_indexes = np.logical_and(
            np.any(data[:, 1:4] != 0, axis=1), np.any(data[:, 4:7] != 0, axis=1)
        )
        selected_indexes = np.logical_not(selected_indexes)
        return selected_indexes

    def _filter_undetected_eyes(self, data: np.ndarray) -> np.ndarray:
        selected_indexes = self._get_indexes_of_undetected_eyes(data)
        return data[selected_indexes]

    def evaluate_on_labeled_data(self):
        y = self._labeled_data[:, -1]
        preds = self.predict(self._labeled_data)[:, 1]
        score = (preds == y).sum()
        print(f'got right {score}/{len(y)} ~ {score/len(y) * 100}%')
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import time

class VisualizerModule:

    def _setup_window(self, video_capture1: cv2.VideoCapture, video_capture2: cv2.VideoCapture):
        # Get video properties
        width1 = int(video_capture1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height1 = int(video_capture1.get(cv2.CAP_PROP_FRAME_HEIGHT))

        width2 = int(video_capture2.get(cv2.CAP_PROP_FRAME_WIDTH))
        height2 = int(video_capture2.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Determine the maximum height between the two videos
        max_height = max(height1, height2)

        # Determine the combined width of the two videos
        combined_width = width1 + width2

        # Create an output window
        cv2.namedWindow('Preview', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Preview', combined_width, max_height + 50)  # Added space for text

    def _read_frames_at_timestamp(self, video_capture: cv2.VideoCapture, timestamp_ms: float):
        video_capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        return video_capture.read()

    def _pad_frame(self, video_capture: cv2.VideoCapture, frame):
        window_height = cv2.getWindowImageRect('Preview')[3]
        video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        padded_frame = np.zeros((window_height, video_width, 3), dtype=np.uint8)
        padded_frame[:video_height, :] = frame

        return padded_frame

    def _create_label(self, timestamp: float, label_df: pd.DataFrame) -> np.ndarray:
        # Create a label image for each frame with the synchronized timestamp
        label_height = 50  # Assuming label height is 50 pixels
        window_width = cv2.getWindowImageRect('Preview')[2]

        label = np.zeros((label_height, window_width, 3), dtype=np.uint8)
        synchronized_timestamp = int(timestamp)
        class_label = self._get_label_from_df(label_df, synchronized_timestamp)
        if class_label is None:
            class_label = 'N/A'

        cv2.putText(label, f'Time: {synchronized_timestamp} ms, Class: {class_label}',
                (10, int(label_height/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        return label


    def visualize_annotated_videos(self, start_timestamp_ms: float, timestamp_step: float, eye_contact_csv_file_path: Path, video_file_path1: Path, video_file_path2: Path):
        label_df = pd.read_csv(eye_contact_csv_file_path)

        video_capture1 = cv2.VideoCapture(str(video_file_path1))
        video_capture2 = cv2.VideoCapture(str(video_file_path2))
        
        self._setup_window(video_capture1, video_capture2)
        
        current_timestamp_ms = start_timestamp_ms

        while True:
            ret1, frame1 = self._read_frames_at_timestamp(video_capture1, current_timestamp_ms)
            ret2, frame2 = self._read_frames_at_timestamp(video_capture2, current_timestamp_ms)
            
            # Break the loop if either video ends
            if not (ret1 and ret2):
                break

            padded_frame1 = self._pad_frame(video_capture1, frame1)
            padded_frame2 = self._pad_frame(video_capture2, frame2)
            
            label = self._create_label(current_timestamp_ms, label_df)
            combined_frame = np.vstack((np.hstack((padded_frame1, padded_frame2)), label))
            
            cv2.imshow('Preview', combined_frame)
            if cv2.waitKey(1000) & 0xFF == ord('q') or  cv2.getWindowProperty('Preview', cv2.WND_PROP_VISIBLE) < 1:
                break

            current_timestamp_ms += timestamp_step

        # Release the video capture objects and close the window
        video_capture1.release()
        video_capture2.release()
        cv2.destroyAllWindows()


    def _get_label_from_df(self, label_df, timestamp):
        filtered_df = label_df[label_df['t'] <= timestamp]
        if filtered_df.empty:
            return None
        else:
            label = filtered_df.iloc[filtered_df['t'].idxmax()]['class']
            if label == 1:
                label = 'eye-contact'
            elif label == 2:
                label = 'no-eye-contact'
            elif label == 3:
                label = 'eyes-not-detected'
            return label
            
        

# Example usage
v = VisualizerModule()
p1 = Path('D:/ML/EyeC/PostProcessing/data/Demos/Demos/Day1/demo1_person1.MOD')
p2 = Path('D:/ML/EyeC/PostProcessing/data/Demos/Demos/Day1/demo1_person2.MOD')
label_p = Path('D:\ML\EyeC\PostProcessing\output\eye_contact_detection.csv')
v.visualize_annotated_videos(0, 330, label_p, p1, p2)
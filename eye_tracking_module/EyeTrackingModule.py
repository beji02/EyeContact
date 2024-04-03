import subprocess
import os

# Change the paths based on your setup
openvino_setup_path = 'C:\Program Files (x86)\Intel\openvino_2022\setupvars'
demo_path = 'D:\ML\EyeC\TryToRun\OpenVINO\ZOO\open_model_zoo\demos\intel64\Release'
opencv_setup_path = 'D:\ML\EyeC\TryToRun\OpenVINO\opencv-4.9.0-windows\opencv\\build\setup_vars_opencv4'
demo_command = 'gaze_estimation_demo -h'


commands = [
    openvino_setup_path,
    'echo Hello',
    'dir',  # You can replace this with your actual commands
    'echo Goodbye'
]

# Join the commands into a single string separated by '&&' to run them sequentially
cmd_string = ' && '.join(commands)

# Run the commands in a new CMD terminal
subprocess.run(['cmd', '/c', cmd_string], shell=True)


gaze_estimation_demo 
-d GPU 
-i D:\ML\EyeC\PostProcessing\data\Demos\Demos\Cuts\demo1_person2_cut.mp4
-m D:\ML\EyeC\TryToRun\OpenVINO\ZOO\open_model_zoo\demos\gaze_estimation_demo\cpp\intel\gaze-estimation-adas-0002\FP16-INT8\gaze-estimation-adas-0002.xml 
-m_fd D:\ML\EyeC\TryToRun\OpenVINO\ZOO\open_model_zoo\demos\gaze_estimation_demo\cpp\intel\face-detection-retail-0004\FP16-INT8\face-detection-retail-0004.xml 
-m_hp D:\ML\EyeC\TryToRun\OpenVINO\ZOO\open_model_zoo\demos\gaze_estimation_demo\cpp\intel\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001.xml 
-m_lm D:\ML\EyeC\TryToRun\OpenVINO\ZOO\open_model_zoo\demos\gaze_estimation_demo\cpp\intel\facial-landmarks-35-adas-0002\FP16-INT8\facial-landmarks-35-adas-0002.xml 
-m_es D:\ML\EyeC\TryToRun\OpenVINO\ZOO\open_model_zoo\demos\gaze_estimation_demo\cpp\public\open-closed-eye-0001\FP32\open-closed-eye-0001.xml 
-r -o D:\ML\EyeC\TryToRun\OpenVINO\ZOO\open_model_zoo\demos\gaze_estimation_demo\mine\output_2.avi > D:\ML\EyeC\PostProcessing\output\log_file_cut_person2


gaze_estimation_demo -d GPU -i D:\ML\EyeC\PostProcessing\data\Demos\Demos\Cuts\demo1_person2_cut.mp4 -m D:\ML\EyeC\TryToRun\OpenVINO\ZOO\open_model_zoo\demos\gaze_estimation_demo\cpp\intel\gaze-estimation-adas-0002\FP16-INT8\gaze-estimation-adas-0002.xml -m_fd D:\ML\EyeC\TryToRun\OpenVINO\ZOO\open_model_zoo\demos\gaze_estimation_demo\cpp\intel\face-detection-retail-0004\FP16-INT8\face-detection-retail-0004.xml -m_hp D:\ML\EyeC\TryToRun\OpenVINO\ZOO\open_model_zoo\demos\gaze_estimation_demo\cpp\intel\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001.xml -m_lm D:\ML\EyeC\TryToRun\OpenVINO\ZOO\open_model_zoo\demos\gaze_estimation_demo\cpp\intel\facial-landmarks-35-adas-0002\FP16-INT8\facial-landmarks-35-adas-0002.xml -m_es D:\ML\EyeC\TryToRun\OpenVINO\ZOO\open_model_zoo\demos\gaze_estimation_demo\cpp\public\open-closed-eye-0001\FP32\open-closed-eye-0001.xml -r -o D:\ML\EyeC\TryToRun\OpenVINO\ZOO\open_model_zoo\demos\gaze_estimation_demo\mine\output_2.avi > D:\ML\EyeC\PostProcessing\output\log_file_cut_person2



# 1. Open CMD
# subprocess.run(['dir && C:'], shell=True)

# # 2. Run openvino setup
# subprocess.run([openvino_setup_path], shell=True)

# # # 3. Change to D:
# subprocess.run(['D:'], shell=True)

# # # 4. Change directory to the demo path
# os.chdir(demo_path)

# # 5. Run opencv setup
# subprocess.run([opencv_setup_path], shell=True)

# # 6. Run the demo command
# subprocess.run([demo_command], shell=True)
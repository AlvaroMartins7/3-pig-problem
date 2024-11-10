import os
import yaml
import argparse
import cv2
import datetime as dt


# extracts each frame from the video
def extract_frames(out_path, video_path):
	cap = cv2.VideoCapture(video_path)
	frame_count = 0

	while True:
		ret, frame = cap.read()
		if not ret:
			break

		frame_filename = os.path.join(out_path, f'frame_{frame_count:04d}.jpg')
		cv2.imwrite(frame_filename, frame)
		frame_count += 1

	cap.release()
	print(f"Extração completa! {frame_count} frames foram salvos em '{out_path}'")
	

# Creates folders to images and annotations
def create_folder(out_path):

	actual_dt = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	out_path = os.path.join(out_path, actual_dt)
	frame_path = os.path.join(out_path, 'frames')

	os.makedirs(frame_path)

	return frame_path


# loads configuration from
def load_config_from_yaml(file_path):

    with open(file_path, 'r') as file:

        return yaml.safe_load(file)


# Parses Arguments
def parse_args():

	parser = argparse.ArgumentParser(description='Generate a dataset with templates.')

	parser.add_argument("--config", dest='config_path', type=str, required=True,
					 	help="Path to YAML configuration file.")
	parser.add_argument("--video", dest='video_name', type=str, required=True,
					 	help="Video name.")

	args = parser.parse_args()
	return args


if __name__ == "__main__":

	args = parse_args()
	config_path = args.config_path
	video_name = args.video_name

	config = load_config_from_yaml(config_path)
	video_path = config['video_path']
	output_path = config['output_path']
      
	out_path = create_folder(output_path)

	extract_frames(out_path, os.path.join(video_path, video_name))
    
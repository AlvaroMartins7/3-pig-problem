import os
import yaml
import argparse
import cv2
import datetime as dt


def extract_frames(out_path, video_path):
	"""Extract every frame from a video file and save them as individual JPEG images.

	Args:
		out_path: Directory where extracted frame images will be saved.
		video_path: Full path to the source video file.
	"""
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
	print(f"Extraction complete! {frame_count} frames saved to '{out_path}'")
	

def create_folder(out_path):
	"""Create a timestamped output directory structure for extracted frames.

	Returns the path to the 'frames' subdirectory.
	"""
	actual_dt = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	out_path = os.path.join(out_path, actual_dt)
	frame_path = os.path.join(out_path, 'frames')

	os.makedirs(frame_path)

	return frame_path


def load_config_from_yaml(file_path):
	"""Load and return configuration parameters from a YAML file."""

	with open(file_path, 'r') as file:

		return yaml.safe_load(file)


def parse_args():
	"""Parse command-line arguments for frame extraction."""

	parser = argparse.ArgumentParser(description='Extract frames from a video file.')

	parser.add_argument("--config", dest='config_path', type=str, required=True,
					 	help="Path to YAML configuration file.")
	parser.add_argument("--video", dest='video_name', type=str, required=True,
					 	help="Video filename (must exist inside the configured video_path).")

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
    
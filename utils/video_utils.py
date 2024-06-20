import cv2
import numpy as np

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()

def split_frame_into_grids(frame, num_x, num_y):
    h, w = frame.shape[:2]
    grid_h, grid_w = h // num_y, w // num_x
    grids = []
    for i in range(num_y):
        for j in range(num_x):
            grids.append(frame[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w])
    return grids

def combine_grids_into_frame(grids, num_x, num_y):
    grid_h, grid_w = grids[0].shape[:2]
    frame_h, frame_w = grid_h * num_y, grid_w * num_x
    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    for i in range(num_y):
        for j in range(num_x):
            frame[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w] = grids[i*num_x + j]
    return frame
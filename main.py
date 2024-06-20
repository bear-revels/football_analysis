from utils import read_video, save_video, SpeedAndDistance_Estimator, Tracker, TeamAssigner, PlayerBallAssigner, CameraMovementEstimator, ViewTransformer
import cv2
import numpy as np
import os

def main():
    # List video files in the input folder
    input_folder = 'files/input_videos'
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    if not video_files:
        print("No video files found in the input folder.")
        return

    # Display the video files to the user
    print("Select a video file to process:")
    for idx, file in enumerate(video_files):
        print(f"{idx + 1}. {file}")

    # Get user input
    selected_idx = int(input("Enter the number corresponding to the video file: ")) - 1
    if selected_idx < 0 or selected_idx >= len(video_files):
        print("Invalid selection.")
        return

    print('Loading the video...')
    selected_file = video_files[selected_idx]
    input_path = os.path.join(input_folder, selected_file)
    output_path = os.path.join('files/output_videos', f"{os.path.splitext(selected_file)[0]}_processed.avi")

    # Read Video
    video_frames = read_video(input_path)

    # Initialize Tracker
    tracker = Tracker('files/models/trackbox_yolov8x_v3.pt')
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='files/stubs/track_stubs.pkl')
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='files/stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Initialize team_ball_control as an array of -1
    team_ball_control = [-1] * len(video_frames)

    # Interpolate Ball Positions if ball tracks are present and not empty
    if "ball" in tracks and any(tracks["ball"]):
        tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

        # Assign Ball Acquisition
        print('Assigning ball possession...')
        player_assigner = PlayerBallAssigner()
        for frame_num, player_track in enumerate(tracks['players']):
            ball_bbox = tracks['ball'][frame_num][1]['bbox']
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

            if assigned_player != -1:
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                team_ball_control[frame_num] = tracks['players'][frame_num][assigned_player]['team']
            else:
                team_ball_control[frame_num] = team_ball_control[frame_num - 1] if frame_num > 0 else -1

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Draw output 
    ## Draw object Tracks
    print('Applying object detection and tracking...')
    output_video_frames = tracker.draw_annotations(video_frames, tracks) #add team_ball_control back as a parameter

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save video
    print('Saving...')
    save_video(output_video_frames, output_path)
    print(f'Complete! The file has been saved to {output_path}')

if __name__ == '__main__':
    main()
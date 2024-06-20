# import numpy as np 
# import cv2

# class ViewTransformer():
#     def __init__(self):
#         court_width = 68
#         court_length = 23.32

#         self.pixel_vertices = np.array([[110, 1035], 
#                                [265, 275], 
#                                [910, 260], 
#                                [1640, 915]])
        
#         self.target_vertices = np.array([
#             [0,court_width],
#             [0, 0],
#             [court_length, 0],
#             [court_length, court_width]
#         ])

#         self.pixel_vertices = self.pixel_vertices.astype(np.float32)
#         self.target_vertices = self.target_vertices.astype(np.float32)

#         self.persepctive_trasnformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

#     def transform_point(self,point):
#         p = (int(point[0]),int(point[1]))
#         is_inside = cv2.pointPolygonTest(self.pixel_vertices,p,False) >= 0 
#         if not is_inside:
#             return None

#         reshaped_point = point.reshape(-1,1,2).astype(np.float32)
#         tranform_point = cv2.perspectiveTransform(reshaped_point,self.persepctive_trasnformer)
#         return tranform_point.reshape(-1,2)

#     def add_transformed_position_to_tracks(self,tracks):
#         for object, object_tracks in tracks.items():
#             for frame_num, track in enumerate(object_tracks):
#                 for track_id, track_info in track.items():
#                     position = track_info['position_adjusted']
#                     position = np.array(position)
#                     position_trasnformed = self.transform_point(position)
#                     if position_trasnformed is not None:
#                         position_trasnformed = position_trasnformed.squeeze().tolist()
#                     tracks[object][frame_num][track_id]['position_transformed'] = position_trasnformed

import numpy as np
import cv2

class ViewTransformer:
    def __init__(self):
        # Soccer field dimensions in meters
        self.court_width = 68  # Width of a standard soccer field
        self.court_length = 105  # Length of a standard soccer field

        # Pixel coordinates of field markers in the video frame
        self.pixel_vertices = np.array([
            [1914, 1027],  # Bottom center
            [1930, 156],   # Top center
            [200, 635],    # Bottom left
            [1043, 249],   # Top left
            [2810, 249],   # Top right
            [3662, 635]    # Bottom right
        ], dtype=np.float32)

        # Corresponding real-world coordinates in meters (assuming a standard soccer field)
        self.target_vertices = np.array([
            [self.court_length / 2, self.court_width],  # Bottom center
            [self.court_length / 2, 0],                 # Top center
            [0, self.court_width],                      # Bottom left
            [0, 0],                                     # Top left
            [self.court_length, 0],                     # Top right
            [self.court_length, self.court_width]       # Bottom right
        ], dtype=np.float32)

        # Compute the homography matrix using all points and RANSAC
        self.perspective_transformer, _ = cv2.findHomography(self.pixel_vertices, self.target_vertices, cv2.RANSAC)

    def transform_point(self, point):
        # Check if point is within the convex hull of the pixel vertices
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None

        # Transform the point using the homography matrix
        reshaped_point = np.array([[point]], dtype=np.float32)
        transformed_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        return transformed_point[0][0]

    def add_transformed_position_to_tracks(self, tracks):
        for object_id, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    transformed_position = self.transform_point(position)
                    if transformed_position is not None:
                        transformed_position = transformed_position.tolist()
                    tracks[object_id][frame_num][track_id]['position_transformed'] = transformed_position

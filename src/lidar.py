import numpy as np
import cv2

class Lidar():

    def __init__(self, map, angle, min_dist, max_dist, pix_size, num_points=360) -> None:
        # Initialize variables
        self.map = map
        self.angle = angle
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.scale = pix_size
        self.num_points = num_points

        self.accuracy = 100 # Number of samples per ray for colision detection
        self.point_size = 1 # Size of points on the map

    def scan(self, point, map, rot=0):
        
        scan_angle = self.angle / self.num_points
        offset_angle = (((np.pi * 2) / scan_angle) - self.num_points) / 2

        distances = []
        for i in range(self.num_points):
            # Create laser point at max distance
            x = np.round(np.sin((scan_angle * (i + offset_angle)) + rot) * self.max_dist * self.scale[0]).astype(np.int64) + point[0]
            y = np.round(np.cos((scan_angle * (i + offset_angle)) + rot) * self.max_dist * self.scale[1]).astype(np.int64) + point[1]

            # Create points samples along rays
            lin_x = np.linspace(point[0], x, self.accuracy)
            lin_y = np.linspace(point[1], y, self.accuracy)

            # if i == 220:
            #     for l_x, l_y in zip(lin_x, lin_y):
            #         if l_x < self.map.shape[1] and l_y < self.map.shape[0]:
            #             if self.map[l_y, l_x, 0] < 255:
            #                 map = cv2.rectangle(map, (l_x - 1, l_y - 1), (l_x + 1, l_y + 1), (0, 0, 255), -1)
            #                 x = l_x
            #                 y = l_y

            #                 break
            #             else:
            #                 map = cv2.rectangle(map, (l_x - 1, l_y - 1), (l_x + 1, l_y + 1), (0, 255, 0), -1)

            # Check for possible collisions
            for l_x, l_y in zip(lin_x, lin_y):
                l_x = np.floor(l_x).astype(np.int64)
                l_y = np.floor(l_y).astype(np.int64)
                if l_x < self.map.shape[1] and l_y < self.map.shape[0]:
                    if self.map[l_y, l_x, 0] < 255:
                        x = l_x
                        y = l_y

                        break
            
            # Calculate the distance od the point
            distances.append(np.hypot((point[0] - x) / self.scale[0], (point[1] - y) / self.scale[1]))

            # Draw points on the map
            map = cv2.rectangle(map, (x - self.point_size, y - self.point_size), (x + self.point_size, y + self.point_size), (0, 0, 255), -1)

        return map, distances
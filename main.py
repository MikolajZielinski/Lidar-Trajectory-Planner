import numpy as np
import pandas as pd
import cv2

class Lidar():

    def __init__(self, map, angle, min_dist, max_dist, pix_size, num_points=360) -> None:
        self.map = map
        self.angle = angle
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.scale = pix_size
        self.num_points = num_points

        self.accuracy = 1000

    def scan(self, point, map, rot=0):
        
        scan_angle = self.angle / self.num_points
        offset_angle = (((np.pi * 2) / scan_angle) - self.num_points) / 2

        for i in range(self.num_points):
            x = np.round(np.sin((scan_angle * (i + offset_angle)) + rot) * self.max_dist * self.scale[0]).astype(np.int64) + point[0]
            y = np.round(np.cos((scan_angle * (i + offset_angle)) + rot) * self.max_dist * self.scale[1]).astype(np.int64) + point[1]

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

            for l_x, l_y in zip(lin_x, lin_y):
                l_x = np.floor(l_x).astype(np.int64)
                l_y = np.floor(l_y).astype(np.int64)
                if l_x < self.map.shape[1] and l_y < self.map.shape[0]:
                    if self.map[l_y, l_x, 0] < 255:
                        x = l_x
                        y = l_y

                        break

            size = 1
            map = cv2.rectangle(map, (x - size, y - size), (x + size, y + size), (0, 0, 255), -1)
        return map

if __name__ == '__main__':
    map = cv2.imread('/home/mikolaj/autoware_map/race_track_01/map.pgm')
    h, w, c = map.shape
    scale = 3
    map = cv2.resize(map, (w * scale, h * scale))
    _, map = cv2.threshold(map, 250, 255, cv2.THRESH_BINARY)
    map_oryg = np.copy(map)
    
    ref_traj = pd.read_csv('/home/mikolaj/autoware_map/race_track_01/trajectory.csv', sep='; ')
    ref_traj = ref_traj[['x_m', 'y_m']]

    offset = (190 * scale, 255 * scale)
    pix_size = (20.5 * scale, 20.5 * scale)

    ref_traj['x_px'] = np.round((ref_traj['x_m'] * pix_size[0]) + offset[0], decimals=0).astype(np.int64)
    ref_traj['y_px'] = np.round((ref_traj['y_m'] * pix_size[0] * -1) + offset[1], decimals=0).astype(np.int64)

    # Start point
    start_point = np.array([250 * scale, 255 * scale])
    start_angle = -np.pi / 2
    step = 0.1
    angle_step = np.pi / 18

    # Draw reference trajectory
    size = 2
    for x, y in ref_traj[['x_px', 'y_px']].itertuples(index=False):
        map_oryg = cv2.rectangle(map_oryg, (x - size, y - size), (x + size, y + size), (255, 0, 255), -1)

    cv2.namedWindow('map')
    while True:
        # Calc start direction
        start_direction = np.array([np.sin(-start_angle), np.cos(start_angle)])

        # Draw bolid
        map = map_oryg.copy()
        cv2.rectangle(map, start_point - 3, start_point + 3, (0, 255, 0), -1)
        cv2.line(map, start_point, ((start_point[0] + (start_direction[0] * 6)).astype(np.int64),  (start_point[1] - (start_direction[1] * 6)).astype(np.int64)), (255, 0, 0), 3)

        # Create lidar
        lid = Lidar(map_oryg, np.pi * 1.5, 0, 5, pix_size)
        map = lid.scan(start_point, map, rot=start_angle)

        # Visualize map
        cv2.imshow('map', map)
        
        # Calc movement
        key = cv2.waitKey(50)
        print(key)
        if key == ord('q'):
            break
        if key == ord('w'):
            start_point[0] = start_point[0] + (step * pix_size[0] * start_direction[0])
            start_point[1] = start_point[1] - (step * pix_size[1] * start_direction[1])
        if key == ord('s'):
            start_point[0] = start_point[0] - (step * pix_size[0] * start_direction[0])
            start_point[1] = start_point[1] + (step * pix_size[1] * start_direction[1])
        if key == ord('a'):
            start_angle = start_angle + angle_step
        if key == ord('d'):
            start_angle = start_angle - angle_step

        
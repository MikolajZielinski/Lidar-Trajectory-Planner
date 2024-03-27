import numpy as np
import pandas as pd
import cv2

from lidar import Lidar

if __name__ == '__main__':

    # Load map file
    map = cv2.imread('/home/mikolaj/autoware_map/race_track_01/map.pgm')
    h, w, c = map.shape
    scale = 3
    map = cv2.resize(map, (w * scale, h * scale))
    _, map = cv2.threshold(map, 250, 255, cv2.THRESH_BINARY)
    map_oryg = np.copy(map)
    
    # Load reference trajectory
    ref_traj = pd.read_csv('/home/mikolaj/autoware_map/race_track_01/trajectory.csv', sep='; ')
    ref_traj = ref_traj[['x_m', 'y_m']]

    # Offset variables to match reference trajectory with map
    offset = (190 * scale, 255 * scale)
    pix_size = (20.5 * scale, 20.5 * scale)

    ref_traj['x_px'] = np.round((ref_traj['x_m'] * pix_size[0]) + offset[0], decimals=0).astype(np.int64)
    ref_traj['y_px'] = np.round((ref_traj['y_m'] * pix_size[0] * -1) + offset[1], decimals=0).astype(np.int64)

    # Start conditions
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
        view_angle = np.pi * 1.5
        num_points = 360
        min_dist = 0
        max_dist = 5
        lid = Lidar(map_oryg, view_angle, min_dist, max_dist, pix_size, num_points=num_points)
        map, distances = lid.scan(start_point, map, rot=start_angle)
        
        scan_angle = view_angle / num_points
        offset_angle = (((np.pi * 2) / scan_angle) - num_points) / 2

        # Convert lidar distances to x, y coordinates
        coords = []
        for i, dist in enumerate(distances):
            if dist < max_dist - 0.05:
                x = (np.sin((scan_angle * (i + offset_angle)) + start_angle) * dist) + (start_point[0] / pix_size[0])
                y = (np.cos((scan_angle * (i + offset_angle)) + start_angle) * dist) + (start_point[1] / pix_size[1])

                coords.append([x, y])
                # # Draw points on the map
                # point_size = 1
                # map = cv2.rectangle(map, (int(x * pix_size[0]) - point_size, int(y * pix_size[1]) - point_size), (int(x * pix_size[0]) + point_size, int(y * pix_size[1]) + point_size), (0, 255, 0), -1)

        # Create point sections 
        line_sections = []
        line_subsection = []
        threshold = 0.3
        for i in range(len(coords)):
            j = i + 1

            if j == len(coords):
                if len(line_subsection) > 10:
                    line_sections.append(line_subsection)
                break

            points_dist = np.hypot(coords[i][0] - coords[j][0], coords[i][0] - coords[j][0])
            if points_dist < threshold:
                line_subsection.append((coords[i], points_dist))
            
            else:
                if len(line_subsection) > 10:
                    line_sections.append(line_subsection)
                
                line_subsection = []
        
        

        # print(a)

        # Visualize map
        cv2.imshow('map', map)
        
        # Calc movement
        key = cv2.waitKey(1)

        # Exit simulation
        if key == ord('q'):
            break

        # Move forward
        if key == ord('w'):
            start_point[0] = start_point[0] + (step * pix_size[0] * start_direction[0])
            start_point[1] = start_point[1] - (step * pix_size[1] * start_direction[1])

        # Move backward
        if key == ord('s'):
            start_point[0] = start_point[0] - (step * pix_size[0] * start_direction[0])
            start_point[1] = start_point[1] + (step * pix_size[1] * start_direction[1])

        # Turn left
        if key == ord('a'):
            start_angle = start_angle + angle_step

        # Turn right
        if key == ord('d'):
            start_angle = start_angle - angle_step

        
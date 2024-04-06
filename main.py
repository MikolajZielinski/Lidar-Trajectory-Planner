import numpy as np
import pandas as pd
import cv2

from lidar import Lidar
from bezier import bezier_curve


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

            points_dist = np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            if points_dist < threshold:
                line_subsection.append((coords[i], points_dist))
            
            else:
                if len(line_subsection) > 10:
                    line_sections.append(line_subsection)
                
                line_subsection = []
        
        # Reduce number of points on a section
        line_sections_reduced = []
        for line_sub in line_sections:
            distance = 0
            line_subsection_reduced = []
            for points, dist in line_sub:
                distance += dist

                if distance >= 0.3:
                    distance = 0
                    line_subsection_reduced.append(points)

            if len(line_subsection_reduced) > 1:
                line_subsection_reduced.insert(0, line_sub[0][0])
                line_sections_reduced.append(line_subsection_reduced)

        # Draw sections
        for line_sub_red in line_sections_reduced:
            for i in range(len(line_sub_red) - 1):
                x1, y1 = line_sub_red[i]
                x1, y1 = int(x1 * pix_size[0]), int(y1 * pix_size[1])
                x2, y2 = line_sub_red[i + 1]
                x2, y2 = int(x2 * pix_size[0]), int(y2 * pix_size[1])

                cv2.line(map, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Calculate normal vectors
        distance_from_border = 0.8
        normal_vectors_sections = []
        for line_sub_red in line_sections_reduced:

            normal_vectors_subsections = []
            for i in range(len(line_sub_red) - 1):
                x1, y1 = line_sub_red[i]
                x2, y2 = line_sub_red[i + 1]

                n_x, n_y = (y2 - y1), -(x2 - x1)
                n_x, n_y = n_x / np.linalg.norm((n_x, n_y)) * distance_from_border, n_y / np.linalg.norm((n_x, n_y)) * distance_from_border

                normal_vectors_subsections.append((n_x + x1, n_y + y1))

                # Draw normal vectors
                x1, y1 = int(x1 * pix_size[0]), int(y1 * pix_size[1])
                x2, y2 = int(x2 * pix_size[0]), int(y2 * pix_size[1])
                n_x, n_y = int(n_x * pix_size[0]), int(n_y * pix_size[1])
                cv2.line(map, (x1, y1), (n_x + x1, n_y + y1), (0, 255, 0), 1)

            normal_vectors_sections.append(normal_vectors_subsections)
                
        # Check if normal vectors are intersecting each other and remove them
        no_inter_sections = []
        for line_sub_red, norm_sub in zip(line_sections_reduced, normal_vectors_sections):
            no_inter_subsections = []
            for i in range(len(norm_sub) - 1):
                x1, y1 = line_sub_red[i]
                x2, y2 = norm_sub[i]
                x3, y3 = line_sub_red[i + 1]
                x4, y4 = norm_sub[i + 1]

                if ((x1 < x3 and x2 > x4) or (x1 > x3 and x2 < x4)) or ((y1 < y3 and y2 > y4) or (y1 > y3 and y2 < y4)):
                    pass

                else:
                    no_inter_subsections.append(norm_sub[i])

            no_inter_subsections.append(norm_sub[-1])

            if len(no_inter_subsections) > 1:
                no_inter_sections.append(no_inter_subsections)

        # Draw normal vector sections
        for norm_sub in no_inter_sections:
            for i in range(len(norm_sub) - 1):
                x1, y1 = norm_sub[i]
                x1, y1 = int(x1 * pix_size[0]), int(y1 * pix_size[1])
                x2, y2 = norm_sub[i + 1]
                x2, y2 = int(x2 * pix_size[0]), int(y2 * pix_size[1])

                cv2.line(map, (x1, y1), (x2, y2), (0, 165, 255), 1)
        
        curve = np.array(bezier_curve(no_inter_sections[0], nTimes=100)).T

        for point in curve:
            x, y = point
            x, y = int(x * pix_size[0]), int(y * pix_size[1])

            cv2.rectangle(map, (x, y), (x, y), (255, 0, 255))
        print(curve)

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

        
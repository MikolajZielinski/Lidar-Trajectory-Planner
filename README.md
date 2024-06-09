# Lidar Trajectory Planner

This is repository with my solution for planning trajectory for f1teenth bolid with lidar data. This is very simplified environment. Only for testing concepts and visualization.

Maps and datapath are taken from [Autoware](https://github.com/autowarefoundation/autoware)

## Running the demo

Simply run

```bash
python3 main.py
```

The window should pop up

![Example map](images/demo_map.png)

Now you can control the bolid (green square with blue arrow) with:
- w - move forward
- a/d - turn left/right
- s - move backwards

Big pink dots are representing the reference trajectory, red dots are lidar points. Two trajectories are planned:
- small blue dots - on intersections always turn left
- small pink dots - on intersections always turn right

## Algorithm

Connect all lidar points together. If the distance between consecutive points is larger then the thresholdand, divide the line into sections. Then reduce number of points in each section.

![Step 1](images/step_1.png)

Find normal vectors to the points.

![Step 2](images/step_2.png)

Connect the ends of normal vecotrs together.

![Step 3](images/step_3.png)

Use bezier curve to smooth obtained path.

![Step 4](images/step_4.png)

Add the ability to propoerly join two path sections. This is done by finding two closest points on both paths and connecting them together. At the same time rest of the points is ignored.

![Step 5](images/step_5.png)

Next step is to recognize the intersections. To achieve that, every start and end point from each subsection is connected with each other. Then the triangle with biggest perimeter is found.

![Step 6](images/step_6.png)

Left corner of this triangle is showing the path to the left, and right corner to the right. Here we can see two examples of trajectories planned on intersections.

![Step 7](images/step_7.png)
![Step 8](images/step_8.png)

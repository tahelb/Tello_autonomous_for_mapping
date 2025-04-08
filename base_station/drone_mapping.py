#drone_mapping.py

# --- Mapping Drone: Imports and Initialization ---
import copy
import json
import numpy as np
import matplotlib
from numpy.ma.extras import average
matplotlib.use("TkAgg")  # Use the TkAgg backend for GUI-based graph display
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import datetime
import random

# --- Constants ---
MAX_KNOWN_DISTANCE = 200  # Obstacles are considered detectable if within 200 cm
UNKNOWN_DISTANCE = 220    # A measured distance of 0 is interpreted as "no detection" and assigned 220 cm
GRID_SIZE = 500           # The generated map will be 500x500 cells, representing a 5x5 meter area
START_POSITION = (250, 380)  # The starting position is defined at (250, 380) in the grid (approximately the center)
VELOCITY = 25             # The default movement speed of the drone in cm/s (used for path estimation)

# --- Generate timestamp-based filename suffix ---
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
date = timestamp  # Used as a unique suffix for generated files

# --- Directional opposites mapping ---
# This dictionary is used to determine the reverse direction of a given movement,
# aiding in path corrections, reverse navigation, and spatial orientation.
direction_opposites = {
    "forward": "backward",
    "backward": "forward",
    "left": "right",
    "right": "left",
    "hover": "hover"  # In the case of hovering, no directional change is needed
}


# --- Fixing JSON files ---
def fix_json_file(input_file, output_file):
    """
    This function is used to correct improperly structured JSON logs
    generated during drone operation. It ensures the format is valid
    by inserting commas between adjacent JSON objects and wrapping them
    in array brackets if needed. The fixed result is saved in a new file.
    """
    try:
        with open(input_file, "r") as file:
            content = file.read()

        # Insert commas between adjacent object blocks
        content = content.replace("}\n{", "},\n{")
        content = content.strip()

        # Wrap with brackets if the content is not in a JSON array
        if not content.startswith("["):
            content = "[" + content
        if not content.endswith("]"):
            content = content + "]"

        # Attempt to decode and re-save the structured JSON content
        data = json.loads(content)
        with open(output_file, "w") as fixed_file:
            json.dump(data, fixed_file, indent=4)

        print(f"The file has been fixed and saved as {output_file}")
        return True

    except json.JSONDecodeError as e:
        print(f"JSON encoding error: {e}")
        return False
    except FileNotFoundError:
        print(f"The file {input_file} was not found!")
        return False


# --- Load structured flight data for analysis ---
def load_flight_data(file_name):
    """
    Loads and parses a structured JSON file containing flight data.
    Returns the decoded list of records, or an empty list in case of failure.
    """
    try:
        with open(file_name, "r") as file:
            return json.load(file)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading flight data: {e}")
        return []


# --- Basic Map Generation: Draws original and recall drone paths with detected walls and unknown regions ---
def generate_map(data, data1, figure_title="Trajectory with Walls and Filled Areas", figure_num=1):
    """
    This function generates a dual-panel plot representing the drone's original flight trajectory
    and its recall path. The walls detected on each side (left/right/forward/backward) are reconstructed
    using the proximity sensor data, and areas with unknown obstacles are marked accordingly.
    """
    # Create side-by-side subplots for original and recall paths
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(40, 20))

    # --- Initialize data structures for original path ---
    trajectory = []  # The full trajectory path of the drone (purple line)
    right_wall = [[]]  # List of segments for walls detected on the right side
    left_wall = [[]]  # List of segments for walls detected on the left side
    forward_wall = [[]]  # List of segments for front-facing walls
    backward_wall = [[]]  # List of segments for rear walls
    unknown_points = []  # Sensor readings beyond known range (visualized in green)

    prev_time = data[0]["time"]  # Initial time for time delta calculation
    prev_x, prev_y = 0, 0  # Starting position (0,0) is the assumed origin
    last_direction = "hover"  # Direction of previous sample (used to handle turns)

    # --- Iterate through each movement sample in the flight log ---
    for sample in data:
        current_time = sample["time"]
        direction = sample["current_direction"]
        distances = sample["distance"]
        delta_time = current_time - prev_time

        if direction == "hover":
            prev_time = current_time
            continue

        # --- Update position based on current direction and time delta ---
        x, y = prev_x, prev_y
        if direction == "forward":
            y += VELOCITY * delta_time
        elif direction == "backward":
            y -= VELOCITY * delta_time
        elif direction == "right":
            x += VELOCITY * delta_time
        elif direction == "left":
            x -= VELOCITY * delta_time

    # --- When direction changes, store wall samples based on last movement ---
        if direction != last_direction:
            if last_direction in ["forward", "backward"]:
                left_wall[-1].append((x - distances["left"], y))
                right_wall[-1].append((x + distances["right"], y))
                forward_wall[-1].append((prev_x, prev_y + distances["forward"]))
                backward_wall[-1].append((prev_x, prev_y - distances["backward"]))
                right_wall.append([])
                left_wall.append([])
            elif last_direction in ["left", "right"]:
                forward_wall[-1].append((x, y + distances["forward"]))
                backward_wall[-1].append((x, y - distances["backward"]))
                left_wall[-1].append((prev_x - distances["left"], prev_y))
                right_wall[-1].append((prev_x + distances["right"], prev_y))
                forward_wall.append([])
                backward_wall.append([])


        # --- Append current position to trajectory ---
        trajectory.append((x, y))

        # --- Update side wall points and mark unknowns if distance > MAX_KNOWN_DISTANCE ---
        if direction in ["forward", "backward"]:
            if distances["left"] < MAX_KNOWN_DISTANCE:
                left_wall[-1].append((x - distances["left"], y))
            else:
                left_wall[-1].append((x - distances["left"], y))
                unknown_points.append((x - distances["left"], y))

            if distances["right"] < MAX_KNOWN_DISTANCE:
                right_wall[-1].append((x + distances["right"], y))
            else:
                right_wall[-1].append((x + distances["right"], y))
                unknown_points.append((x + distances["right"], y))

        elif direction in ["left", "right"]:
            if distances["forward"] < MAX_KNOWN_DISTANCE:
                forward_wall[-1].append((x, y + distances["forward"]))
            else:
                forward_wall[-1].append((x, y + distances["forward"]))
                unknown_points.append((x, y + distances["forward"]))

            if distances["backward"] < MAX_KNOWN_DISTANCE:
                backward_wall[-1].append((x, y - distances["backward"]))
            else:
                backward_wall[-1].append((x, y - distances["backward"]))
                unknown_points.append((x, y - distances["backward"]))

        prev_time = current_time
        prev_x, prev_y = x, y
        last_direction = direction


    # --- Plot trajectory line ---
    if trajectory:
        x_vals, y_vals = zip(*trajectory)
        ax1.plot(x_vals, y_vals, marker="o", linestyle="-", color="purple", markersize=4, label="Trajectory")

    # --- Fill known areas between side walls (yellow regions) ---
    yellow_polygons_right_left = []
    yellow_polygons_forward_backward = []

    for left, right in zip(left_wall, right_wall):
        if len(left) > 1 and len(right) > 1:
            lx, ly = zip(*left)
            rx, ry = zip(*right)
            vertices = list(zip(lx, ly)) + list(zip(rx[::-1], ry[::-1]))
            polygon = mpath.Path(vertices)
            yellow_polygons_right_left.append(polygon)
            ax1.fill_betweenx(ly, lx, rx, color="yellow", alpha=0.4)

    for front, back in zip(forward_wall, backward_wall):
        if len(front) > 1 and len(back) > 1:
            fx, fy = zip(*front)
            bx, by = zip(*back)
            vertices = list(zip(fx, fy)) + list(zip(bx[::-1], by[::-1]))
            polygon = mpath.Path(vertices)
            yellow_polygons_forward_backward.append(polygon)
            ax1.fill_between(fx, by, fy, color="yellow", alpha=0.4)

    # --- Internal utility functions ---
    def is_point_inside_polygon(polygon_type, point, polygons_r_l, polygons_f_b):
        """ Checks whether a point lies within a defined polygonal region. """
        x, y = point
        if polygon_type == "right_left":
            for polygon in polygons_r_l:
                if polygon.contains_point((x, y), radius=-1e-10):
                    return True
        else:
            for polygon in polygons_f_b:
                if polygon.contains_point((x, y), radius=-1e-10):
                    return True
        return False

    def remove_internal_points(polygon_type, wall, polygons_rl, polygons_fb):
        """ Removes points from wall segments that lie within filled polygon areas. """
        return [point for point in wall if not is_point_inside_polygon(polygon_type, point, polygons_rl, polygons_fb)]

    def remove_unknown_points(wall, unknown_points):
        """ Filters out segments that pass through unknown regions. """
        cleaned_walls = []
        current_segment = []
        for point in wall:
            if point in unknown_points and current_segment:
                if len(current_segment) > 1:
                    cleaned_walls.append(current_segment)
                current_segment = []
            else:
                current_segment.append(point)
        if current_segment and len(current_segment) > 1:
            cleaned_walls.append(current_segment)
        return cleaned_walls


    # --- Final processing of walls for display: remove overlapping and unknown points ---
    cleaned_forward_wall = [remove_internal_points("right_left", segment, yellow_polygons_right_left, yellow_polygons_forward_backward) for segment in forward_wall if len(segment) > 1]
    finish_forward_wall = []
    for segment in cleaned_forward_wall:
        finish_forward_wall.extend(remove_unknown_points(segment, unknown_points))

    cleaned_backward_wall = [remove_internal_points("right_left", segment, yellow_polygons_right_left, yellow_polygons_forward_backward) for segment in backward_wall if len(segment) > 1]
    finish_backward_wall = []
    for segment in cleaned_backward_wall:
        finish_backward_wall.extend(remove_unknown_points(segment, unknown_points))

    cleaned_left_wall = [remove_internal_points("forward_backward", segment, yellow_polygons_right_left, yellow_polygons_forward_backward) for segment in left_wall if len(segment) > 1]
    finish_left_wall = []
    for segment in cleaned_left_wall:
        finish_left_wall.extend(remove_unknown_points(segment, unknown_points))

    cleaned_right_wall = [remove_internal_points("forward_backward", segment, yellow_polygons_right_left, yellow_polygons_forward_backward) for segment in right_wall if len(segment) > 1]
    finish_right_wall = []
    for segment in cleaned_right_wall:
        finish_right_wall.extend(remove_unknown_points(segment, unknown_points))

    # --- Draw wall lines ---
    for wall in finish_right_wall:
        if len(wall) > 1:
            w_x, w_y = zip(*wall)
            ax1.plot(w_x, w_y, linestyle="-", color="black", linewidth=2)

    for wall in finish_left_wall:
        if len(wall) > 1:
            w_x, w_y = zip(*wall)
            ax1.plot(w_x, w_y, linestyle="-", color="black", linewidth=2)

    for wall in finish_forward_wall:
        if len(wall) > 1:
            w_x, w_y = zip(*wall)
            ax1.plot(w_x, w_y, linestyle="-", color="black", linewidth=2)


    for wall in finish_backward_wall:
        if len(wall) > 1:
            w_x, w_y = zip(*wall)
            ax1.plot(w_x, w_y, linestyle="-", color="black", linewidth=2)

    # --- Draw unknown points (green dots) ---
    if unknown_points:
        u_x, u_y = zip(*unknown_points)
        ax1.scatter(u_x, u_y, color="green", marker="o", s=30, label="Unknown Area")

    # --- Final plot formatting ---
    ax1.set_xlim(-1100, 1100)
    ax1.set_ylim(-1100, 1100)
    ax1.set_xlabel("X Position (cm)")
    ax1.set_ylabel("Y Position (cm)")
    ax1.set_title(f"{figure_title} {date}")
    ax1.legend()
    ax1.grid(True)


    # --- Initialize data structures for recall path (RECALL) ---
    trajectory1 = []  # The drone's recall trajectory (green path)
    right_wall1 = [[]]  # Detected right wall segments during recall
    left_wall1 = [[]]  # Detected left wall segments during recall
    forward_wall1 = [[]]  # Detected front wall segments during recall
    backward_wall1 = [[]]  # Detected rear wall segments during recall
    unknown_points1 = []  # Points with distance data beyond known range

    prev_time1 = data1[0]["time"]  # Initialize previous time for time delta computation
    prev_x1, prev_y1 = prev_x, prev_y  # Start recall path from the same position as original end
    last_direction1 = "hover"  # Last direction used for wall estimation

    # --- Iterate through recall data samples ---
    for sample1 in data1:
        current_time1 = sample1["time"]
        direction1 = sample1["current_direction"]
        distances1 = sample1["distance"]
        delta_time1 = current_time1 - prev_time1

        if direction1 == "hover":
            prev_time1 = current_time1
            continue

        # --- Update coordinates based on direction and delta time ---
        x1, y1 = prev_x1, prev_y1
        if direction1 == "forward":
            y1 += VELOCITY * delta_time1
        elif direction1 == "backward":
            y1 -= VELOCITY * delta_time1
        elif direction1 == "right":
            x1 += VELOCITY * delta_time1
        elif direction1 == "left":
            x1 -= VELOCITY * delta_time1

        # --- Record wall geometry when direction changes ---
        if direction1 != last_direction1:
            if last_direction1 in ["forward", "backward"]:
                left_wall1[-1].append((x1 - distances1["left"], y1))
                right_wall1[-1].append((x1 + distances1["right"], y1))
                forward_wall1[-1].append((prev_x1, prev_y1 + distances1["forward"]))
                backward_wall1[-1].append((prev_x1, prev_y1 - distances1["backward"]))
                right_wall1.append([])
                left_wall1.append([])
            elif last_direction1 in ["left", "right"]:
                forward_wall1[-1].append((x1, y1 + distances1["forward"]))
                backward_wall1[-1].append((x1, y1 - distances1["backward"]))
                left_wall1[-1].append((prev_x1 - distances1["left"], prev_y1))
                right_wall1[-1].append((prev_x1 + distances1["right"], prev_y1))
                forward_wall1.append([])
                backward_wall1.append([])

        # --- Save current recall position ---
        trajectory1.append((x1, y1))

        # --- Estimate walls and unknown points based on recall movement ---
        if direction1 in ["forward", "backward"]:
            if distances1["left"] < MAX_KNOWN_DISTANCE:
                left_wall1[-1].append((x1 - distances1["left"], y1))
            else:
                left_wall1[-1].append((x1 - distances1["left"], y1))
                unknown_points1.append((x1 - distances1["left"], y1))

            if distances1["right"] < MAX_KNOWN_DISTANCE:
                right_wall1[-1].append((x1 + distances1["right"], y1))
            else:
                right_wall1[-1].append((x1 + distances1["right"], y1))
                unknown_points1.append((x1 + distances1["right"], y1))

        elif direction1 in ["left", "right"]:
            if distances1["forward"] < MAX_KNOWN_DISTANCE:
                forward_wall1[-1].append((x1, y1 + distances1["forward"]))
            else:
                forward_wall1[-1].append((x1, y1 + distances1["forward"]))
                unknown_points1.append((x1, y1 + distances1["forward"]))

            if distances1["backward"] < MAX_KNOWN_DISTANCE:
                backward_wall1[-1].append((x1, y1 - distances1["backward"]))
            else:
                backward_wall1[-1].append((x1, y1 - distances1["backward"]))
                unknown_points1.append((x1, y1 - distances1["backward"]))

        prev_time1 = current_time1
        prev_x1, prev_y1 = x1, y1
        last_direction1 = direction1

    # --- Plot recall trajectory (green line) ---
    if trajectory1:
        x_vals1, y_vals1 = zip(*trajectory1)
        ax2.plot(x_vals1, y_vals1, marker="o", linestyle="-", color="green", markersize=4, label="Recall")

    # --- Fill yellow areas between walls for RECALL path ---
    yellow_polygons_right_left1 = []
    yellow_polygons_forward_backward1 = []

    for left1, right1 in zip(left_wall1, right_wall1):
        if len(left1) > 1 and len(right1) > 1:
            lx1, ly1 = zip(*left1)
            rx1, ry1 = zip(*right1)
            vertices1 = list(zip(lx1, ly1)) + list(zip(rx1[::-1], ry1[::-1]))
            polygon1 = mpath.Path(vertices1)
            yellow_polygons_right_left1.append(polygon1)
            ax2.fill_betweenx(ly1, lx1, rx1, color="yellow", alpha=0.4)

    for front1, back1 in zip(forward_wall1, backward_wall1):
        if len(front1) > 1 and len(back1) > 1:
            fx1, fy1 = zip(*front1)
            bx1, by1 = zip(*back1)
            vertices1 = list(zip(fx1, fy1)) + list(zip(bx1[::-1], by1[::-1]))
            polygon1 = mpath.Path(vertices1)
            yellow_polygons_forward_backward1.append(polygon1)
            ax2.fill_between(fx1, by1, fy1, color="yellow", alpha=0.4)

    # ---  filtering functions for RECALL wall cleanup ---
    def is_point_inside_polygon(polygon_type1, point1, polygons_r_l1, polygons_f_b1):
        x1, y1 = point1
        if polygon_type1 == "right_left":
            for polygon in polygons_r_l1:
                if polygon.contains_point((x1, y1), radius=-1e-10):
                    return True
        else:
            for polygon in polygons_f_b1:
                if polygon.contains_point((x1, y1), radius=-1e-10):
                    return True
        return False

    def remove_internal_points(polygon_type1, wall1, polygons_rl1, polygons_fb1):
        return [point1 for point1 in wall1 if
                not is_point_inside_polygon(polygon_type1, point1, polygons_rl1, polygons_fb1)]

    def remove_unknown_points(wall1, unknow_points1):
        cleaned_walls1 = []
        current_segment1 = []
        for point1 in wall1:
            if point1 in unknow_points1 and current_segment1:
                if len(current_segment1) > 1:
                    cleaned_walls1.append(current_segment1)
                current_segment1 = []
            else:
                current_segment1.append(point1)
        if current_segment1 and len(current_segment1) > 1:
            cleaned_walls1.append(current_segment1)
        return cleaned_walls1

    # --- Process RECALL wall data ---
    cleaned_forward_wall1 = [remove_internal_points("right_left", segment1, yellow_polygons_right_left1, yellow_polygons_forward_backward1) for segment1 in forward_wall1 if len(segment1) > 1]
    finish_forward_wall1 = []
    for segment1 in cleaned_forward_wall1:
        finish_forward_wall1.extend(remove_unknown_points(segment1, unknown_points1))

    cleaned_backward_wall1 = [remove_internal_points("right_left", segment1, yellow_polygons_right_left1, yellow_polygons_forward_backward1) for segment1 in backward_wall1 if len(segment1) > 1]
    finish_backward_wall1 = []
    for segment1 in cleaned_backward_wall1:
        finish_backward_wall1.extend(remove_unknown_points(segment1, unknown_points1))

    cleaned_left_wall1 = [remove_internal_points("forward_backward", segment1, yellow_polygons_right_left1, yellow_polygons_forward_backward1) for segment1 in left_wall1 if len(segment1) > 1]
    finish_left_wall1 = []
    for segment1 in cleaned_left_wall1:
        finish_left_wall1.extend(remove_unknown_points(segment1, unknown_points1))

    cleaned_right_wall1 = [remove_internal_points("forward_backward", segment1, yellow_polygons_right_left1, yellow_polygons_forward_backward1) for segment1 in right_wall1 if len(segment1) > 1]
    finish_right_wall1 = []
    for segment1 in cleaned_right_wall1:
        finish_right_wall1.extend(remove_unknown_points(segment1, unknown_points1))

    # --- Draw RECALL wall segments ---
    for wall1 in finish_right_wall1:
        if len(wall1) > 1:
            w_x1, w_y1 = zip(*wall1)
            ax2.plot(w_x1, w_y1, linestyle="-", color="black", linewidth=2)

    for wall1 in finish_left_wall1:
        if len(wall1) > 1:
            w_x1, w_y1 = zip(*wall1)
            ax2.plot(w_x1, w_y1, linestyle="-", color="black", linewidth=2)

    for wall1 in finish_forward_wall1:
        if len(wall1) > 1:
            w_x1, w_y1 = zip(*wall1)
            ax2.plot(w_x1, w_y1, linestyle="-", color="black", linewidth=2)

    for wall1 in finish_backward_wall1:
        if len(wall1) > 1:
            w_x1, w_y1 = zip(*wall1)
            ax2.plot(w_x1, w_y1, linestyle="-", color="black", linewidth=2)

    # --- Plot unknown zones for RECALL ---
    if unknown_points1:
        u_x1, u_y1 = zip(*unknown_points1)
        ax2.scatter(u_x1, u_y1, color="green", marker="o", s=30, label="Unknown Area")

    # --- Formatting for RECALL plot ---
    ax2.set_xlim(-1100, 1100)
    ax2.set_ylim(-1100, 1100)
    ax2.set_xlabel("X Position (cm)")
    ax2.set_ylabel("Y Position (cm)")
    ax2.set_title(f"Recall {date}")
    ax2.legend()
    ax2.grid(True)

    # --- Display the complete visualization ---
    plt.show()


# --- Generate Combined Map: Original and Recall Trajectories ---
def generate_combine_map(data, data1, figure_title="Trajectory with Walls and Filled Areas", figure_num=1):
    plt.figure(figure_num, figsize=(20, 20))
    """
    Generate a combined visualization of both the original and recall drone trajectories,
    including surrounding wall detections and unknown regions. The function overlays both 
    paths to visually compare discrepancies and analyze trajectory consistency.
    """

    # --- Initialize data structures for original trajectory ---
    trajectory2 = []  # Midpoints between original and recall trajectories
    trajectory = []  # The path traveled by the drone during original exploration (purple)
    right_wall = [[]]  # Right-side obstacle points
    left_wall = [[]]   # Left-side obstacle points
    forward_wall = [[]]  # Forward obstacle points
    backward_wall = [[]]  # Backward obstacle points
    unknown_points = []  # Points with unreliable or out-of-range sensor values (green)

    # Initialize positional and temporal parameters
    prev_time = data[0]["time"]
    prev_x, prev_y = 0, 0
    last_direction = "hover"

    # --- Process the original path data to reconstruct trajectory and map obstacles ---
    for sample in data:
        current_time = sample["time"]
        direction = sample["current_direction"]
        distances = sample["distance"]
        delta_time = current_time - prev_time

        if direction == "hover":
            prev_time = current_time
            continue

        # Update drone's position based on velocity and direction
        x, y = prev_x, prev_y
        if direction == "forward":
            y += VELOCITY * delta_time
        elif direction == "backward":
            y -= VELOCITY * delta_time
        elif direction == "right":
            x += VELOCITY * delta_time
        elif direction == "left":
            x -= VELOCITY * delta_time

        # If direction changed, close previous wall segments
        if direction != last_direction:
            if last_direction in ["forward", "backward"]:
                left_wall[-1].append((x - distances["left"], y))
                right_wall[-1].append((x + distances["right"], y))
                forward_wall[-1].append((prev_x, prev_y + distances["forward"]))
                backward_wall[-1].append((prev_x, prev_y - distances["backward"]))
                right_wall.append([])
                left_wall.append([])
            elif last_direction in ["left", "right"]:
                forward_wall[-1].append((x, y + distances["forward"]))
                backward_wall[-1].append((x, y - distances["backward"]))
                left_wall[-1].append((prev_x - distances["left"], prev_y))
                right_wall[-1].append((prev_x + distances["right"], prev_y))
                forward_wall.append([])
                backward_wall.append([])

        # Append current position to the trajectory
        trajectory.append((x, y))

        # Append wall points and mark unknowns
        if direction in ["forward", "backward"]:
            if distances["left"] < MAX_KNOWN_DISTANCE:
                left_wall[-1].append((x - distances["left"], y))
            else:
                left_wall[-1].append((x - distances["left"], y))
                unknown_points.append((x - distances["left"], y))

            if distances["right"] < MAX_KNOWN_DISTANCE:
                right_wall[-1].append((x + distances["right"], y))
            else:
                right_wall[-1].append((x + distances["right"], y))
                unknown_points.append((x + distances["right"], y))

        elif direction in ["left", "right"]:
            if distances["forward"] < MAX_KNOWN_DISTANCE:
                forward_wall[-1].append((x, y + distances["forward"]))
            else:
                forward_wall[-1].append((x, y + distances["forward"]))
                unknown_points.append((x, y + distances["forward"]))

            if distances["backward"] < MAX_KNOWN_DISTANCE:
                backward_wall[-1].append((x, y - distances["backward"]))
            else:
                backward_wall[-1].append((x, y - distances["backward"]))
                unknown_points.append((x, y - distances["backward"]))

        # Update for next iteration
        prev_time = current_time
        prev_x, prev_y = x, y
        last_direction = direction

    # --- Plot original trajectory ---
    if trajectory:
        x_vals, y_vals = zip(*trajectory)
        plt.plot(x_vals, y_vals, marker="o", linestyle="-", color="purple", markersize=4, label="Trajectory")

    # --- Fill between left and right walls to show explored area ---
    yellow_polygons_right_left = []
    yellow_polygons_forward_backward = []

    for left, right in zip(left_wall, right_wall):
        if len(left) > 1 and len(right) > 1:
            lx, ly = zip(*left)
            rx, ry = zip(*right)
            vertices = list(zip(lx, ly)) + list(zip(rx[::-1], ry[::-1]))
            polygon = mpath.Path(vertices)
            yellow_polygons_right_left.append(polygon)
            plt.fill_betweenx(ly, lx, rx, color="yellow", alpha=0.4)

    for front, back in zip(forward_wall, backward_wall):
        if len(front) > 1 and len(back) > 1:
            fx, fy = zip(*front)
            bx, by = zip(*back)
            vertices = list(zip(fx, fy)) + list(zip(bx[::-1], by[::-1]))
            polygon = mpath.Path(vertices)
            yellow_polygons_forward_backward.append(polygon)
            plt.fill_between(fx, by, fy, color="yellow", alpha=0.4)

    # --- Utility functions to clean up noise and overlap ---
    def is_point_inside_polygon(polygon_type, point, polygons_r_l, polygons_f_b):
        x, y = point
        if polygon_type == "right_left":
            for polygon in polygons_r_l:
                if polygon.contains_point((x, y), radius=-1e-10):
                    return True
        else:
            for polygon in polygons_f_b:
                if polygon.contains_point((x, y), radius=-1e-10):
                    return True
        return False

    def remove_internal_points(polygon_type, wall, polygons_rl, polygons_fb):
        return [point for point in wall if not is_point_inside_polygon(polygon_type, point, polygons_rl, polygons_fb)]

    def remove_unknown_points(wall, unknown_points):
        cleaned_walls = []
        current_segment = []
        for point in wall:
            if point in unknown_points and current_segment:
                if len(current_segment) > 1:
                    cleaned_walls.append(current_segment)
                current_segment = []
            else:
                current_segment.append(point)
        if current_segment and len(current_segment) > 1:
            cleaned_walls.append(current_segment)
        return cleaned_walls

    # --- Process and draw all four wall directions ---
    cleaned_forward_wall = [remove_internal_points("right_left", segment, yellow_polygons_right_left, yellow_polygons_forward_backward) for segment in forward_wall if len(segment) > 1]
    finish_forward_wall = []
    for segment in cleaned_forward_wall:
        finish_forward_wall.extend(remove_unknown_points(segment, unknown_points))

    cleaned_backward_wall = [remove_internal_points("right_left", segment, yellow_polygons_right_left, yellow_polygons_forward_backward) for segment in backward_wall if len(segment) > 1]
    finish_backward_wall = []
    for segment in cleaned_backward_wall:
        finish_backward_wall.extend(remove_unknown_points(segment, unknown_points))

    cleaned_left_wall = [remove_internal_points("forward_backward", segment, yellow_polygons_right_left, yellow_polygons_forward_backward) for segment in left_wall if len(segment) > 1]
    finish_left_wall = []
    for segment in cleaned_left_wall:
        finish_left_wall.extend(remove_unknown_points(segment, unknown_points))

    cleaned_right_wall = [remove_internal_points("forward_backward", segment, yellow_polygons_right_left, yellow_polygons_forward_backward) for segment in right_wall if len(segment) > 1]
    finish_right_wall = []
    for segment in cleaned_right_wall:
        finish_right_wall.extend(remove_unknown_points(segment, unknown_points))

    for wall in finish_right_wall:
        if len(wall) > 1:
            w_x, w_y = zip(*wall)
            plt.plot(w_x, w_y, linestyle="-", color="black", linewidth=2)

    for wall in finish_left_wall:
        if len(wall) > 1:
            w_x, w_y = zip(*wall)
            plt.plot(w_x, w_y, linestyle="-", color="black", linewidth=2)

    for wall in finish_forward_wall:
        if len(wall) > 1:
            w_x, w_y = zip(*wall)
            plt.plot(w_x, w_y, linestyle="-", color="black", linewidth=2)

    for wall in finish_backward_wall:
        if len(wall) > 1:
            w_x, w_y = zip(*wall)
            plt.plot(w_x, w_y, linestyle="-", color="black", linewidth=2)

    # --- Draw unknown points (in green) ---
    if unknown_points:
        u_x, u_y = zip(*unknown_points)
        plt.scatter(u_x, u_y, color="green", marker="o", s=30, label="Unknown Area")

    # --- Final formatting and display ---
    plt.xlim(-1100, 1100)
    plt.ylim(-1100, 1100)
    plt.xlabel("X Position (cm)")
    plt.ylabel("Y Position (cm)")
    plt.title(f"{figure_title} {date}")
    plt.legend()
    plt.grid(True)

    # --- Initialize data structures for recall trajectory ---
    trajectory1 = []  # Path of the drone during the recall phase (green)
    right_wall1 = [[]]  # Right-side obstacle points
    left_wall1 = [[]]  # Left-side obstacle points
    forward_wall1 = [[]]  # Forward obstacle points
    backward_wall1 = [[]]  # Backward obstacle points
    unknown_points1 = []  # Points where sensor data was unreliable or exceeded range

    # Start from the last position of the original path
    prev_time1 = data1[0]["time"]
    prev_x1, prev_y1 = prev_x, prev_y
    last_direction1 = "hover"

    # --- Process the recall path data ---
    for sample1 in data1:
        current_time1 = sample1["time"]
        direction1 = sample1["current_direction"]
        distances1 = sample1["distance"]
        delta_time1 = current_time1 - prev_time1

        if direction1 == "hover":
            prev_time1 = current_time1
            continue

        x1, y1 = prev_x1, prev_y1
        if direction1 == "forward":
            y1 += VELOCITY * delta_time1
        elif direction1 == "backward":
            y1 -= VELOCITY * delta_time1
        elif direction1 == "right":
            x1 += VELOCITY * delta_time1
        elif direction1 == "left":
            x1 -= VELOCITY * delta_time1

        # Close and reset wall segments on direction change
        if direction1 != last_direction1:
            if last_direction1 in ["forward", "backward"]:
                left_wall1[-1].append((x1 - distances1["left"], y1))
                right_wall1[-1].append((x1 + distances1["right"], y1))
                forward_wall1[-1].append((prev_x1, prev_y1 + distances1["forward"]))
                backward_wall1[-1].append((prev_x1, prev_y1 - distances1["backward"]))
                right_wall1.append([])
                left_wall1.append([])
            elif last_direction1 in ["left", "right"]:
                forward_wall1[-1].append((x1, y1 + distances1["forward"]))
                backward_wall1[-1].append((x1, y1 - distances1["backward"]))
                left_wall1[-1].append((prev_x1 - distances1["left"], prev_y1))
                right_wall1[-1].append((prev_x1 + distances1["right"], prev_y1))
                forward_wall1.append([])
                backward_wall1.append([])

        trajectory1.append((x1, y1))

        # Append wall points and track unknowns
        if direction1 in ["forward", "backward"]:
            if distances1["left"] < MAX_KNOWN_DISTANCE:
                left_wall1[-1].append((x1 - distances1["left"], y1))
            else:
                left_wall1[-1].append((x1 - distances1["left"], y1))
                unknown_points1.append((x1 - distances1["left"], y1))

            if distances1["right"] < MAX_KNOWN_DISTANCE:
                right_wall1[-1].append((x1 + distances1["right"], y1))
            else:
                right_wall1[-1].append((x1 + distances1["right"], y1))
                unknown_points1.append((x1 + distances1["right"], y1))

        elif direction1 in ["left", "right"]:
            if distances1["forward"] < MAX_KNOWN_DISTANCE:
                forward_wall1[-1].append((x1, y1 + distances1["forward"]))
            else:
                forward_wall1[-1].append((x1, y1 + distances1["forward"]))
                unknown_points1.append((x1, y1 + distances1["forward"]))

            if distances1["backward"] < MAX_KNOWN_DISTANCE:
                backward_wall1[-1].append((x1, y1 - distances1["backward"]))
            else:
                backward_wall1[-1].append((x1, y1 - distances1["backward"]))
                unknown_points1.append((x1, y1 - distances1["backward"]))

        prev_time1 = current_time1
        prev_x1, prev_y1 = x1, y1
        last_direction1 = direction1

    # --- Plot recall trajectory ---
    if trajectory1:
        x_vals1, y_vals1 = zip(*trajectory1)
        plt.plot(x_vals1, y_vals1, marker="o", linestyle="-", color="green", markersize=4, label="Recall")

    # --- Fill areas between walls for the recall path ---
    yellow_polygons_right_left1 = []
    yellow_polygons_forward_backward1 = []

    for left1, right1 in zip(left_wall1, right_wall1):
        if len(left1) > 1 and len(right1) > 1:
            lx1, ly1 = zip(*left1)
            rx1, ry1 = zip(*right1)
            vertices1 = list(zip(lx1, ly1)) + list(zip(rx1[::-1], ry1[::-1]))
            polygon1 = mpath.Path(vertices1)
            yellow_polygons_right_left1.append(polygon1)
            plt.fill_betweenx(ly1, lx1, rx1, color="yellow", alpha=0.4)

    for front1, back1 in zip(forward_wall1, backward_wall1):
        if len(front1) > 1 and len(back1) > 1:
            fx1, fy1 = zip(*front1)
            bx1, by1 = zip(*back1)
            vertices1 = list(zip(fx1, fy1)) + list(zip(bx1[::-1], by1[::-1]))
            polygon1 = mpath.Path(vertices1)
            yellow_polygons_forward_backward1.append(polygon1)
            plt.fill_between(fx1, by1, fy1, color="yellow", alpha=0.4)

    # --- Clean and draw final wall segments ---
    cleaned_forward_wall1 = [
        remove_internal_points("right_left", segment1, yellow_polygons_right_left1, yellow_polygons_forward_backward1)
        for segment1 in forward_wall1 if len(segment1) > 1]
    finish_forward_wall1 = []
    for segment1 in cleaned_forward_wall1:
        finish_forward_wall1.extend(remove_unknown_points(segment1, unknown_points1))

    cleaned_backward_wall1 = [
        remove_internal_points("right_left", segment1, yellow_polygons_right_left1, yellow_polygons_forward_backward1)
        for segment1 in backward_wall1 if len(segment1) > 1]
    finish_backward_wall1 = []
    for segment1 in cleaned_backward_wall1:
        finish_backward_wall1.extend(remove_unknown_points(segment1, unknown_points1))

    cleaned_left_wall1 = [remove_internal_points("forward_backward", segment1, yellow_polygons_right_left1,
                                                 yellow_polygons_forward_backward1) for segment1 in left_wall1 if
                          len(segment1) > 1]
    finish_left_wall1 = []
    for segment1 in cleaned_left_wall1:
        finish_left_wall1.extend(remove_unknown_points(segment1, unknown_points1))

    cleaned_right_wall1 = [remove_internal_points("forward_backward", segment1, yellow_polygons_right_left1,
                                                  yellow_polygons_forward_backward1) for segment1 in right_wall1 if
                           len(segment1) > 1]
    finish_right_wall1 = []
    for segment1 in cleaned_right_wall1:
        finish_right_wall1.extend(remove_unknown_points(segment1, unknown_points1))

    for wall1 in finish_right_wall1:
        if len(wall1) > 1:
            w_x1, w_y1 = zip(*wall1)
            plt.plot(w_x1, w_y1, linestyle="-", color="black", linewidth=2)

    for wall1 in finish_left_wall1:
        if len(wall1) > 1:
            w_x1, w_y1 = zip(*wall1)
            plt.plot(w_x1, w_y1, linestyle="-", color="black", linewidth=2)

    for wall1 in finish_forward_wall1:
        if len(wall1) > 1:
            w_x1, w_y1 = zip(*wall1)
            plt.plot(w_x1, w_y1, linestyle="-", color="black", linewidth=2)

    for wall1 in finish_backward_wall1:
        if len(wall1) > 1:
            w_x1, w_y1 = zip(*wall1)
            plt.plot(w_x1, w_y1, linestyle="-", color="black", linewidth=2)

    # --- Draw unknown points for recall ---
    if unknown_points1:
        u_x1, u_y1 = zip(*unknown_points1)
        plt.scatter(u_x1, u_y1, color="green", marker="o", s=30, label="Unknown Area")

    # --- Compute and draw average trajectory between original and recall ---
    for i in range(len(trajectory)):
        (x2, y2) = trajectory[i]
        (x3, y3) = trajectory1[len(trajectory1) - i - 1]
        x4 = (x2 + x3) * 0.5
        y4 = (y2 + y3) * 0.5
        trajectory2.append((x4, y4))

    if trajectory2:
        x_vals2, y_vals2 = zip(*trajectory2)
        plt.plot(x_vals2, y_vals2, marker="o", linestyle="-", color="blue", markersize=4, label="Average")

    # --- Final map settings ---
    plt.xlim(-1100, 1100)
    plt.ylim(-1100, 1100)
    plt.xlabel("X Position (cm)")
    plt.ylabel("Y Position (cm)")
    plt.title(f"Combined Map {date}")
    plt.legend()
    plt.grid(True)
    plt.show()


def generate_average_map(data, data1, figure_title="Trajectory with Walls and Filled Areas", figure_num=1):
    """
       Generates a visual map showing the averaged trajectory and estimated surrounding walls
       using original and recall data. It averages corresponding positions and distances
       to create a smoothed and unified representation of the flight path.
    """
    plt.figure(figure_num, figsize=(20, 20))

    distances2 = {"forward": 220, "backward": 220, "left": 220, "right": 220}
    prev_x2, prev_y2 = 0, 0

    trajectory = []


    prev_time = data[0]["time"]
    prev_x, prev_y = 0, 0
    last_direction = "hover"

    trajectory1 = []
    trajectory2 = []
    right_wall2 = [[]]
    left_wall2 = [[]]
    forward_wall2 = [[]]
    backward_wall2 = [[]]
    unknown_points2 = []

    prev_x1, prev_y1 = 0, 0

    # Match sample count for original and recall data
    data1_reversed = data1[::-1]
    min_length = min(len(data), len(data1_reversed))
    data = data[:min_length]
    data1_reversed = data1_reversed[:min_length]
    prev_time1 = data1_reversed[0]["time"]


    for i in range(min_length):
        sample = data[i]
        sample1 = data1_reversed[i]

        # Extract time, direction, and sensor readings
        current_time = sample["time"]
        direction = sample["current_direction"]
        distances = sample["distance"]
        delta_time = current_time - prev_time

        current_time1 = sample1["time"]
        direction1 = sample1["current_direction"]
        distances1 = sample1["distance"]
        delta_time1 = current_time1 - prev_time1

        # Skip 'hover' states (no movement)
        if direction == "hover":
            prev_time = current_time
            continue

        if direction1 == "hover":
            prev_time1 = current_time1
            continue

        # Calculate positions for both directions
        x, y = prev_x, prev_y
        x1, y1 = prev_x1, prev_y1

        # Update coordinates for original flight
        if direction == "forward":
            y += VELOCITY * delta_time
        elif direction == "backward":
            y -= VELOCITY * delta_time
        elif direction == "right":
            x += VELOCITY * delta_time
        elif direction == "left":
            x -= VELOCITY * delta_time

        # Update coordinates for recall flight
        if direction1 == "forward":
            y1 += VELOCITY * delta_time1
        elif direction1 == "backward":
            y1 -= VELOCITY * delta_time1
        elif direction1 == "right":
            x1 += VELOCITY * delta_time1
        elif direction1 == "left":
            x1 -= VELOCITY * delta_time1

        # Average distances for wall estimation
        for direction2 in distances:
            distances2[direction2] = (distances1[direction2] + distances[direction2]) * 0.5

        # Calculate averaged position
        x2 = (x1 + x) * 0.5
        y2 = (y1 + y) * 0.5

        # Detect direction changes and start new wall segments accordingly
        if direction != last_direction:
            if last_direction in ["forward", "backward"]:
                left_wall2[-1].append((x2 - distances2["left"], y2))
                right_wall2[-1].append((x2 + distances2["right"], y2))
                forward_wall2[-1].append((prev_x2, prev_y2 + distances2["forward"]))
                backward_wall2[-1].append((prev_x2, prev_y2 - distances2["backward"]))
                right_wall2.append([])
                left_wall2.append([])
            elif last_direction in ["left", "right"]:
                forward_wall2[-1].append((x2, y2 + distances2["forward"]))
                backward_wall2[-1].append((x2, y2 - distances2["backward"]))
                left_wall2[-1].append((prev_x2 - distances2["left"], prev_y2))
                right_wall2[-1].append((prev_x2 + distances2["right"], prev_y2))
                forward_wall2.append([])
                backward_wall2.append([])

        # Record path points
        trajectory.append((x, y))
        trajectory1.append((x1, y1))
        trajectory2.append((x2, y2))

        # Estimate walls based on distance thresholds
        if direction in ["forward", "backward"]:
            if distances2["left"] < MAX_KNOWN_DISTANCE:
                left_wall2[-1].append((x2 - distances2["left"], y2))
            else:
                left_wall2[-1].append((x2 - distances2["left"], y2))
                unknown_points2.append((x2 - distances2["left"], y2))

            if distances2["right"] < MAX_KNOWN_DISTANCE:
                right_wall2[-1].append((x2 + distances2["right"], y2))
            else:
                right_wall2[-1].append((x2 + distances2["right"], y2))
                unknown_points2.append((x2 + distances2["right"], y2))

        elif direction in ["left", "right"]:
            if distances2["forward"] < MAX_KNOWN_DISTANCE:
                forward_wall2[-1].append((x2, y2 + distances2["forward"]))
            else:
                forward_wall2[-1].append((x2, y2 + distances2["forward"]))
                unknown_points2.append((x2, y2 + distances2["forward"]))

            if distances["backward"] < MAX_KNOWN_DISTANCE:
                backward_wall2[-1].append((x2, y2 - distances2["backward"]))
            else:
                backward_wall2[-1].append((x2, y2 - distances2["backward"]))
                unknown_points2.append((x2, y2 - distances2["backward"]))

        # Update time and position trackers
        prev_time = current_time
        prev_x, prev_y = x, y
        last_direction = direction

        prev_time1 = current_time1
        prev_x1, prev_y1 = x1, y1
        last_direction1 = direction1
        prev_x2, prev_y2 = x2, y2

    # Draw averaged trajectory path
    if trajectory2:
        x_vals2, y_vals2 = zip(*trajectory2)
        plt.plot(x_vals2, y_vals2, marker="o", linestyle="-", color="blue", markersize=4, label="average path")

    # --- Wall Processing and Filtering ---

    # Draw yellow regions between left-right and front-back walls
    yellow_polygons_right_left2 = []
    yellow_polygons_forward_backward2 = []

    for left2, right2 in zip(left_wall2, right_wall2):
        if len(left2) > 1 and len(right2) > 1:
            lx2, ly2 = zip(*left2)
            rx2, ry2 = zip(*right2)
            vertices2 = list(zip(lx2, ly2)) + list(zip(rx2[::-1], ry2[::-1]))
            polygon2 = mpath.Path(vertices2)
            yellow_polygons_right_left2.append(polygon2)
            plt.fill_betweenx(ly2, lx2, rx2, color="yellow", alpha=0.4)


    for front2, back2 in zip(forward_wall2, backward_wall2):
        if len(front2) > 1 and len(back2) > 1:
            fx2, fy2 = zip(*front2)
            bx2, by2 = zip(*back2)
            vertices2 = list(zip(fx2, fy2)) + list(zip(bx2[::-1], by2[::-1]))
            polygon2 = mpath.Path(vertices2)
            yellow_polygons_forward_backward2.append(polygon2)
            plt.fill_between(fx2, by2, fy2, color="yellow", alpha=0.4)


    # פונקציות לבדיקת נקודות פנימיות וסינון
    def is_point_inside_polygon(polygon_type2, point2, polygons_r_l2, polygons_f_b2):
        x2, y2 = point2
        if polygon_type2 == "right_left":
            for polygon in polygons_r_l2:
                if polygon.contains_point((x2, y2), radius=-1e-10):
                    return True
        else:
            for polygon in polygons_f_b2:
                if polygon.contains_point((x2, y2), radius=-1e-10):
                    return True
        return False

    def remove_internal_points(polygon_type2, wall2, polygons_rl2, polygons_fb2):
        return [point2 for point2 in wall2 if not is_point_inside_polygon(polygon_type2, point2, polygons_rl2, polygons_fb2)]

    def remove_unknown_points(wall2, unknow_points2):
        cleaned_walls2 = []
        current_segment2 = []
        for point2 in wall2:
            if point2 in unknow_points2 and current_segment2:
                if len(current_segment2) > 1:
                    cleaned_walls2.append(current_segment2)
                current_segment2 = []
            else:
                current_segment2.append(point2)
        if current_segment2 and len(current_segment2) > 1:
            cleaned_walls2.append(current_segment2)
        return cleaned_walls2

    # --- Clean and draw final wall segments ---
    cleaned_forward_wall2 = [remove_internal_points("right_left", segment2, yellow_polygons_right_left2, yellow_polygons_forward_backward2) for segment2 in forward_wall2 if len(segment2) > 1]
    finish_forward_wall2 = []
    for segment2 in cleaned_forward_wall2:
        finish_forward_wall2.extend(remove_unknown_points(segment2, unknown_points2))

    cleaned_backward_wall2 = [remove_internal_points("right_left", segment2, yellow_polygons_right_left2, yellow_polygons_forward_backward2) for segment2 in backward_wall2 if len(segment2) > 1]
    finish_backward_wall2 = []
    for segment2 in cleaned_backward_wall2:
        finish_backward_wall2.extend(remove_unknown_points(segment2, unknown_points2))

    cleaned_left_wall2 = [remove_internal_points("forward_backward", segment2, yellow_polygons_right_left2, yellow_polygons_forward_backward2) for segment2 in left_wall2 if len(segment2) > 1]
    finish_left_wall2 = []
    for segment2 in cleaned_left_wall2:
        finish_left_wall2.extend(remove_unknown_points(segment2, unknown_points2))

    cleaned_right_wall2 = [remove_internal_points("forward_backward", segment2, yellow_polygons_right_left2, yellow_polygons_forward_backward2) for segment2 in right_wall2 if len(segment2) > 1]
    finish_right_wall2 = []
    for segment2 in cleaned_right_wall2:
        finish_right_wall2.extend(remove_unknown_points(segment2, unknown_points2))

    for wall2 in finish_right_wall2:
        if len(wall2) > 1:
            w_x2, w_y2 = zip(*wall2)
            plt.plot(w_x2, w_y2, linestyle="-", color="black", linewidth=2)


    for wall2 in finish_left_wall2:
        if len(wall2) > 1:
            w_x2, w_y2 = zip(*wall2)
            plt.plot(w_x2, w_y2, linestyle="-", color="black", linewidth=2)


    for wall2 in finish_forward_wall2:
        if len(wall2) > 1:
            w_x2, w_y2 = zip(*wall2)
            plt.plot(w_x2, w_y2, linestyle="-", color="black", linewidth=2)


    for wall2 in finish_backward_wall2:
        if len(wall2) > 1:
            w_x2, w_y2 = zip(*wall2)
            plt.plot(w_x2, w_y2, linestyle="-", color="black", linewidth=2)


    # --- Draw unknown points ---
    if unknown_points2:
        u_x1, u_y1 = zip(*unknown_points2)
        plt.scatter(u_x1, u_y1, color="green", marker="o", s=30, label="Unknown Area")

    # --- Final map settings ---
    plt.xlim(-1100, 1100)
    plt.ylim(-1100, 1100)
    plt.xlabel("X Position (cm)")
    plt.ylabel("Y Position (cm)")
    plt.title(f"average map {date}")
    plt.legend()
    plt.grid(True)
    plt.show()


def generate_revaluation_map(data, data1, figure_title="Trajectory with Walls and Filled Areas", figure_num=1):
    """
    This function generates a refined evaluation map based on two drone flight logs: the original flight and its reverse (recall).
    It calculates average sensor readings and trajectory positions, reconstructs estimated walls, filters internal and noisy points,
    and plots the final corrected wall map and trajectory.
    """
    plt.figure(figure_num, figsize=(20, 20))

    distances2 = {"forward": 220, "backward": 220, "left": 220, "right": 220}
    prev_x2, prev_y2 = 0, 0

    trajectory = []
    prev_time = data[0]["time"]
    prev_x, prev_y = 0, 0
    last_direction = "hover"

    trajectory1 = []
    trajectory2 = []
    right_wall2 = [[]]
    left_wall2 = [[]]
    forward_wall2 = [[]]
    backward_wall2 = [[]]
    unknown_points2 = []
    corrners = []
    corrners_point = []
    right_wall_2 = [[]]
    left_wall_2 = [[]]
    forward_wall_2 = [[]]
    backward_wall_2 = [[]]
    prev_x1, prev_y1 = 0, 0

    # Reverse the recall trajectory to align temporally with the original one
    data1_reversed = data1[::-1]
    min_length = min(len(data), len(data1_reversed))
    data = data[:min_length]
    data1_reversed = data1_reversed[:min_length]
    prev_time1 = data1_reversed[0]["time"]

    # --- Loop over both paths in parallel and process data ---
    for i in range(min_length):
        sample = data[i]
        sample1 = data1_reversed[i]

        # Extract timing and direction info from both logs
        current_time = sample["time"]
        direction = sample["current_direction"]
        distances = sample["distance"]
        delta_time = current_time - prev_time
        current_time1 = sample1["time"]
        direction1 = sample1["current_direction"]
        distances1 = sample1["distance"]
        delta_time1 = current_time1 - prev_time1

        # Skip if drone was hovering (no movement)
        if direction == "hover":
            prev_time = current_time
            continue
        if direction1 == "hover":
            prev_time1 = current_time1
            continue

        # Update positions based on direction and elapsed time
        x, y = prev_x, prev_y
        x1, y1 = prev_x1, prev_y1

        if direction == "forward":
            y += VELOCITY * delta_time
        elif direction == "backward":
            y -= VELOCITY * delta_time
        elif direction == "right":
            x += VELOCITY * delta_time
        elif direction == "left":
            x -= VELOCITY * delta_time

        if direction1 == "forward":
            y1 += VELOCITY * delta_time1
        elif direction1 == "backward":
            y1 -= VELOCITY * delta_time1
        elif direction1 == "right":
            x1 += VELOCITY * delta_time1
        elif direction1 == "left":
            x1 -= VELOCITY * delta_time1

        # --- Average each direction's sensor distance from original and recall ---
        for direction2 in distances:
            distances2[direction2] = (distances1[direction2] + distances[direction2]) * 0.5


        # Compute average position
        x2 = (x1 + x) * 0.5
        y2 = (y1 + y) * 0.5

        if (last_direction == "forward" and direction == "right") or (
                last_direction == "right" and direction == "forward") or (
                last_direction == "left" and direction == "backward") or (
                last_direction == "backward" and direction == "left"):
            x2 += 0

        if direction != last_direction:
            if last_direction in ["forward", "backward"]:
                left_wall_2[-1].append((x2 - distances2["left"], y2))
                right_wall_2[-1].append((x2 + distances2["right"], y2))
                forward_wall_2[-1].append((prev_x2, prev_y2 + distances2["forward"]))
                backward_wall_2[-1].append((prev_x2, prev_y2 - distances2["backward"]))
                right_wall_2.append([])
                left_wall_2.append([])
            elif last_direction in ["left", "right"]:
                forward_wall_2[-1].append((x2, y2 + distances2["forward"]))
                backward_wall_2[-1].append((x2, y2 - distances2["backward"]))
                left_wall_2[-1].append((prev_x2 - distances2["left"], prev_y2))
                right_wall_2[-1].append((prev_x2 + distances2["right"], prev_y2))
                forward_wall_2.append([])
                backward_wall_2.append([])

        trajectory.append((x, y))
        trajectory1.append((x1, y1))
        x2 = (x1 + x) * 0.5
        y2 = (y1 + y) * 0.5
        trajectory2.append((x2, y2))

        if direction in ["forward", "backward"]:
            if distances2["left"] < MAX_KNOWN_DISTANCE:
                left_wall_2[-1].append((x2 - distances2["left"], y2))
            else:
                point = (x2 - distances2["left"], y2)
                left_wall_2[-1].append((x2 - distances2["left"], y2))
                unknown_points2.append((x2 - distances2["left"], y2))
                print(f"append to unknow_points2 {point}")

            if distances2["right"] < MAX_KNOWN_DISTANCE:
                right_wall_2[-1].append((x2 + distances2["right"], y2))
            else:
                right_wall_2[-1].append((x2 + distances2["right"], y2))
                unknown_points2.append((x2 + distances2["right"], y2))
                point = (x2 + distances2["right"], y2)
                print(f"append to unknow_points2 {point}")

        elif direction in ["left", "right"]:
            if distances2["forward"] < MAX_KNOWN_DISTANCE:
                forward_wall_2[-1].append((x2, y2 + distances2["forward"]))
            else:
                forward_wall_2[-1].append((x2, y2 + distances2["forward"]))
                unknown_points2.append((x2, y2 + distances2["forward"]))
                point = (x2, y2 + distances2["forward"])
                print(f"append to unknow_points2 {point}")

            if distances["backward"] < MAX_KNOWN_DISTANCE:
                backward_wall_2[-1].append((x2, y2 - distances2["backward"]))
            else:
                backward_wall_2[-1].append((x2, y2 - distances2["backward"]))
                unknown_points2.append((x2, y2 - distances2["backward"]))
                point = (x2, y2 - distances2["backward"])
                print(f"append to unknow_points2 {point}")

        prev_time = current_time
        prev_x, prev_y = x, y
        last_direction = direction

        prev_time1 = current_time1
        prev_x1, prev_y1 = x1, y1
        prev_x2, prev_y2 = x2, y2


    if trajectory2:
        x_vals2, y_vals2 = zip(*trajectory2)
        plt.plot(x_vals2, y_vals2, marker="o", linestyle="-", color="blue", markersize=4, label="revaluation")

    def average_wall(direction_wall, wall2):
        final_segment2 = []
        corrner_0 = False
        lenths_wall = 0
        average_distance = 0
        if len(wall2) > 1:
            if direction_wall == "forward" or direction_wall == "backward":
                for point2 in wall2:
                    (x_2, y_2) = point2
                    if point2 in unknown_points2:
                        print(f"not include the point in average compute {point2}")
                    else:
                        average_distance += y_2
                        lenths_wall += 1
                        print(f"add to average compute the point {point2}")
                if lenths_wall >= 1:
                    average_distance /= lenths_wall
                for point2 in wall2:
                    (x_2, y_2) = point2
                    real_point = (x_2, average_distance)
                    if point2 in unknown_points2:
                        unknown_points2.remove((x_2, y_2))
                        unknown_points2.append(real_point)
                    else:
                        real_point = (x_2, average_distance)
                    final_segment2.append(real_point)
            else:
                for point2 in wall2:
                    (x_2, y_2) = point2
                    if point2 in unknown_points2:
                        print(f"not include the point in average compute {point2}")
                    else:
                        average_distance += x_2
                        lenths_wall += 1
                        print(f"add to average compute the point {point2}")
                if lenths_wall >= 1:
                    average_distance /= lenths_wall
                for point2 in wall2:
                    (x_2, y_2) = point2
                    real_point = (average_distance, y_2)
                    if point2 not in unknown_points2:
                        real_point = (average_distance, y_2)
                    else:
                        unknown_points2.remove((x_2, y_2))
                        unknown_points2.append(real_point)
                    final_segment2.append(real_point)
        return final_segment2


    for segment2 in forward_wall_2:
         forward_wall2.append(average_wall("forward", segment2))

    for segment2 in backward_wall_2:
         backward_wall2.append(average_wall("backward", segment2))

    for segment2 in left_wall_2:
        left_wall2.append(average_wall("left", segment2))

    for segment2 in right_wall_2:
        right_wall2.append(average_wall("right", segment2))

    for corrner in corrners:
        corrners_point.append(corrner[1])



    yellow_polygons_right_left2 = []
    yellow_polygons_forward_backward2 = []

    for left2, right2 in zip(left_wall2, right_wall2):
        if len(left2) > 1 and len(right2) > 1:
            lx2, ly2 = zip(*left2)
            rx2, ry2 = zip(*right2)
            vertices2 = list(zip(lx2, ly2)) + list(zip(rx2[::-1], ry2[::-1]))
            polygon2 = mpath.Path(vertices2)
            yellow_polygons_right_left2.append(polygon2)
            plt.fill_betweenx(ly2, lx2, rx2, color="yellow", alpha=0.4)


    for front2, back2 in zip(forward_wall2, backward_wall2):
        if len(front2) > 1 and len(back2) > 1:
            fx2, fy2 = zip(*front2)
            bx2, by2 = zip(*back2)
            vertices2 = list(zip(fx2, fy2)) + list(zip(bx2[::-1], by2[::-1]))
            polygon2 = mpath.Path(vertices2)
            yellow_polygons_forward_backward2.append(polygon2)
            plt.fill_between(fx2, by2, fy2, color="yellow", alpha=0.4)



    def is_point_inside_polygon(polygon_type2, point2, polygons_r_l2, polygons_f_b2):
        (x_2, y_2) = point2
        if polygon_type2 == "right_left":
            for polygon in polygons_r_l2:
                if polygon.contains_point((x_2, y_2), radius=-1e-10):
                    return True
        else:
            for polygon in polygons_f_b2:
                if polygon.contains_point((x_2, y_2), radius=-1e-10):
                    return True
        return False

    def remove_internal_points(polygon_type2, wall2, polygons_rl2, polygons_fb2):
        return [point2 for point2 in wall2 if not is_point_inside_polygon(polygon_type2, point2, polygons_rl2, polygons_fb2)]

    def remove_unknown_points(wall2, unknow_points2):
        cleaned_walls2 = []
        current_segment2 = []
        for point2 in wall2:
            if point2 in unknow_points2 and current_segment2:
                if len(current_segment2) > 1:
                    cleaned_walls2.append(current_segment2)
                current_segment2 = []
            else:
                current_segment2.append(point2)
        if current_segment2 and len(current_segment2) > 1:
            cleaned_walls2.append(current_segment2)
        return cleaned_walls2

    def remove_corrners_points(wall2, corrners_point2):
        cleaned_walls2 = []
        current_segment2 = []
        for point2 in wall2:
            if point2 in corrners_point2 and current_segment2:
                if len(current_segment2) > 1:
                    cleaned_walls2.append(current_segment2)
                current_segment2 = []
            else:
                current_segment2.append(point2)
        if current_segment2 and len(current_segment2) > 1:
            cleaned_walls2.append(current_segment2)
        return cleaned_walls2


    cleaned_forward_wall2 = [remove_internal_points("right_left", segment2, yellow_polygons_right_left2, yellow_polygons_forward_backward2) for segment2 in forward_wall2 if len(segment2) > 1]
    finish_forward_wall2 = []
    final_forward_wall2 = []
    for segment2 in cleaned_forward_wall2:
        finish_forward_wall2.extend(remove_unknown_points(segment2, unknown_points2))
    for segment2 in finish_forward_wall2:
        final_forward_wall2.extend(remove_corrners_points(segment2, corrners_point))


    cleaned_backward_wall2 = [remove_internal_points("right_left", segment2, yellow_polygons_right_left2, yellow_polygons_forward_backward2) for segment2 in backward_wall2 if len(segment2) > 1]
    finish_backward_wall2 = []
    final_backward_wall2 = []
    for segment2 in cleaned_backward_wall2:
        finish_backward_wall2.extend(remove_unknown_points(segment2, unknown_points2))
    for segment2 in finish_backward_wall2:
        final_backward_wall2.extend(remove_corrners_points(segment2, corrners_point))


    cleaned_left_wall2 = [remove_internal_points("forward_backward", segment2, yellow_polygons_right_left2, yellow_polygons_forward_backward2) for segment2 in left_wall2 if len(segment2) > 1]
    finish_left_wall2 = []
    final_left_wall2 = []
    for segment2 in cleaned_left_wall2:
        finish_left_wall2.extend(remove_unknown_points(segment2, unknown_points2))
    for segment2 in finish_left_wall2:
        final_left_wall2.extend(remove_corrners_points(segment2, corrners_point))



    cleaned_right_wall2 = [remove_internal_points("forward_backward", segment2, yellow_polygons_right_left2, yellow_polygons_forward_backward2) for segment2 in right_wall2 if len(segment2) > 1]
    finish_right_wall2 = []
    final_right_wall2 = []
    for segment2 in cleaned_right_wall2:
        finish_right_wall2.extend(remove_unknown_points(segment2, unknown_points2))
    for segment2 in finish_right_wall2:
        final_right_wall2.extend(remove_corrners_points(segment2, corrners_point))


    for wall2 in final_right_wall2:
        if len(wall2) > 1:
            w_x2, w_y2 = zip(*wall2)
            plt.plot(w_x2, w_y2, linestyle="-", color="black", linewidth=2)


    for wall2 in final_left_wall2:
        if len(wall2) > 1:
            w_x2, w_y2 = zip(*wall2)
            plt.plot(w_x2, w_y2, linestyle="-", color="black", linewidth=2)


    for wall2 in final_forward_wall2:
        if len(wall2) > 1:
            w_x2, w_y2 = zip(*wall2)
            plt.plot(w_x2, w_y2, linestyle="-", color="black", linewidth=2)


    for wall2 in final_backward_wall2:
        if len(wall2) > 1:
            w_x2, w_y2 = zip(*wall2)
            plt.plot(w_x2, w_y2, linestyle="-", color="black", linewidth=2)


    if unknown_points2:
        u_x1, u_y1 = zip(*unknown_points2)
        plt.scatter(u_x1, u_y1, color="green", marker="o", s=30, label="Unknown Area")

    for corrner in corrners:
        c_x2, c_y2 = zip(*corrner)
        plt.plot(c_x2, c_y2, linestyle="-", color="gray", linewidth=2)


    plt.xlim(-1100, 1100)
    plt.ylim(-1100, 1100)
    plt.xlabel("X Position (cm)")
    plt.ylabel("Y Position (cm)")
    plt.title(f"revaluation map {date}")
    plt.legend()
    plt.grid(True)
    plt.show()

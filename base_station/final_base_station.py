#basestation code
# --- Base Station: Imports and Initialization ---
from djitellopy import Tello
import cv2
import pygame
import numpy as np
import time
import socket
import json
import threading
import logging
import datetime
import subprocess
import matplotlib.pyplot as plt
from drone_mapping import fix_json_file, process_flight_data, load_flight_data
from drone_mapping import generate_map, generate_combine_map, generate_average_map, generate_revaluation_map

# Configure logging system for runtime information
logging.basicConfig(level=logging.INFO)

# --- Constants ---
S = 25  # Drone motion speed in centimeters per second (cm/s)
FPS = 30  # Video streaming frame rate in frames per second
BS_IP = '192.168.2.1'  # Static IP address of the base station
PI_IP = '192.168.2.11'  # Static IP address of the Raspberry Pi (PIE)
PIE_PORT = 12345  # TCP port for drone command/control communication
SERVER_PORT = 65432  # TCP port used for messaging and alerts
FILE_PORT = 65436  # TCP port used for file transfer
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
FILE_NAME = f"flight_data_{timestamp}.json"  # Unique filename for storing session data

ALERT_DISTANCE = 30  # Minimum safety threshold for obstacle detection in cm
keyup_flag = True  # State variable used to detect key release in the control GUI

# --- Global State Variables ---
current_direction = "hover"  # Initial direction set to 'hover'; will be updated dynamically
running = False  # Indicates whether the drone is actively operating
start_time = time.time()  # Timestamp used to measure elapsed flight duration
distances = {
    "forward": 220.0,
    "backward": 220.0,
    "left": 220.0,
    "right": 220.0
}  # Dictionary storing the latest known distances from all directions

travel_times = {}  # Dictionary mapping each direction to the accumulated travel time
travel_log = []  # List storing the chronological path data for analysis and mapping
position = (0, 0)  # Cartesian representation of the drone's current location on a 2D grid

# Dictionary used for resolving opposite directions (e.g., forward <-> backward)
opposites = {
    "forward": "backward",
    "backward": "forward",
    "left": "right",
    "right": "left"
}

opposite_direction = "backward"  # Initial opposite direction used for path correction
distance_back = None  # Placeholder for storing reference distance during recall
DISTANCE_TOLERANCE = 40  # Acceptable threshold for distance deviation (cm) during return flight
last_keep_time = 0  # Timestamp for last received 'keepalive' signal from the drone

# Variables used for storing flight data files received from the drone
original_file = None
recall_file = None

# Print the generated file name for debugging and tracking
print(FILE_NAME)


# --- Mapping Workflow: Fix, Process, and Generate All Visual Maps ---
def process_and_generate_all_maps(original_filename, recall_filename):
    """
    This function orchestrates the entire mapping process by:
    1. Fixing potential inconsistencies in the original and recall JSON files.
    2. Preprocessing the data for consistency and duplication handling.
    3. Generating multiple types of visual maps from the processed trajectory data:
       - Standard directional map
       - Combined trajectory map
       - Average path map
       - Revaluation-based path correction map
    """
    fix_json_file(original_filename, "fixed_original.json")
    fix_json_file(recall_filename, "fixed_recall.json")

    process_flight_data("fixed_original.json")
    process_flight_data("fixed_recall.json")

    original_data = load_flight_data("duplicate_fixed_original.json")
    recall_data = load_flight_data("duplicate_fixed_recall.json")

    if original_data and recall_data:
        generate_map(original_data, recall_data, "Forward Mapping", 1)
        generate_combine_map(original_data, recall_data, "Forward Mapping", 2)
        generate_average_map(original_data, recall_data, "Forward Mapping", 3)
        generate_revaluation_map(original_data, recall_data, "Forward Mapping", 4)

    print("All mapping processes completed!")


# --- File Reception from Raspberry Pi ---
def receive_file():
    """
    This function handles the reception of flight data files from the Raspberry Pi device.
    It listens for incoming TCP connections on FILE_PORT, receives metadata (header),
    and writes the binary contents of the file locally. Once both the original and
    recall flight data files are received, it triggers the mapping process.
    """
    global original_file, recall_file

    file_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    file_socket.bind(('0.0.0.0', FILE_PORT))
    file_socket.listen(5)
    logging.info("Waiting for flight data file from Raspberry Pi...")

    while True:
        file_client_socket, addr = file_socket.accept()
        try:
            # Receive fixed-length header containing file metadata
            header_data = file_client_socket.recv(1024).decode().strip()
            header = json.loads(header_data)
            file_name = header["filename"]
            file_type = header["type"]
        except Exception as e:
            logging.error(f"Failed to parse header: {e}")
            file_client_socket.close()
            continue

        # Receive and write the file content
        with open(file_name, "wb") as file:
            while True:
                file_data = file_client_socket.recv(4096)
                if not file_data:
                    break
                file.write(file_data)
        logging.info(f"Received {file_type} file: {file_name}")
        file_client_socket.send("FILE_RECEIVED".encode())
        file_client_socket.close()

        # Update state variables based on the file type
        if file_type.lower() == "original":
            original_file = file_name
        elif file_type.lower() == "recall":
            recall_file = file_name

        # If both required files have been received, begin map processing
        if original_file and recall_file:
            logging.info("Both original and recall files received. Starting map processing...")
            process_and_generate_all_maps(original_file, recall_file)
            original_file, recall_file = None, None  # Reset for next session





# --- Save real-time flight data to file ---
def save_to_file(distances1, current_direction1):
    """
    This function logs real-time flight data including distance measurements from all directions
    and the current movement direction. The data is written to a local JSON file with a timestamp.
    """
    elapsed_time = round(time.time() - start_time, 2)
    record = {
        "time": elapsed_time,
        "current_direction": current_direction1,
        "distance": distances1,
    }
    with open(FILE_NAME, "a") as file:
        file.write(json.dumps(record, indent=4) + "\n")




# --- Receiving Runtime Data from Raspberry Pi ---
def receive_data():
    """
    This function maintains an open TCP socket connection to receive real-time updates from the Raspberry Pi,
    including:
    - Direction and distance telemetry
    - Obstacle alerts
    - Recall procedure initiation
    - Return path status
    The information is processed, logged, and stored locally.
    """
    global distances, current_direction, running, distance_back

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', SERVER_PORT))
    server_socket.listen(5)
    logging.info("Waiting for data from Raspberry Pi...")

    while True:
        while running:
            try:
                client_socket, addr = server_socket.accept()
                data = client_socket.recv(1024).decode()

                # Handle flight direction and distance update
                if data.startswith("DATA:"):
                    parts = data.split(",")
                    current_direction = parts[0].split("=")[1]
                    distances = {
                        "forward": float(parts[1].split("=")[1]),
                        "backward": float(parts[2].split("=")[1]),
                        "left": float(parts[3].split("=")[1]),
                        "right": float(parts[4].split("=")[1])
                    }
                    save_to_file(distances, current_direction)
                    logging.info(f"Received: {data}")

                # Handle obstacle alert message
                elif data.startswith("ALERT:"):
                    logging.info(f"Alert received: {data}")

                # Handle recall initiation message
                elif data.startswith("START RECALL:"):
                    file_name = data.split("START RECALL:")[1]
                    logging.info(f"start recall, received the name of file: {file_name}")

                # Handle position update during return path
                elif data.startswith("RETURN:"):
                    logging.info(f"the distance in this direction is {data}")

                # Handle recall process termination
                elif data == "END_RECALL":
                    logging.info("Recall completed, landing drone...")
                    print(travel_log)

                else:
                    logging.info("wait for data")

                client_socket.close()
            except Exception as e:
                logging.error(f"Error receiving data: {e}")
                if client_socket:
                    client_socket.close()



# --- Drone Class ---
class Drone:
    """
    This class encapsulates the high-level control of the drone via socket-based commands
    sent to the Raspberry Pi unit. It includes initialization, command transmission,
    movement control, and lifecycle actions such as takeoff, landing, and recall.
    Attributes:
        control_socket (socket): TCP socket for sending control commands.
        speed (int): Default linear speed of the drone in cm/s.
    """
    def __init__(self, stream=False):
        print("Initializing Drone...")
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.control_socket.bind((BS_IP, PIE_PORT))
        self.control_socket.connect((PI_IP, PIE_PORT))
        print("Connection established.")
        self.speed = S
        self.send_command("command")
        self.send_command("streamon")
        self.send_command(f"speed {self.speed}")
        self.stream = stream

    def send_command(self, command):
        """
        Sends a string command to the Raspberry Pi and waits for acknowledgment.

        Args:
            command (str): The command to be transmitted (e.g., 'rc 0 20 0 0').
        """
        try:
            self.control_socket.send(command.encode('utf-8'))
            response = self.control_socket.recv(1024).decode()
            if "OK" not in response:
                logging.error(f"Command {command} failed: {response}")
        except Exception as e:
            logging.error(f"Error sending command: {e}")

    def update(self, left_right, forward_backward, up_down, direction):
        """
        Sends an RC directional command to the drone to update its motion in 3D space.

        Args:
            left_right (int): Lateral motion value (left/right).
            forward_backward (int): Longitudinal motion value (forward/backward).
            up_down (int): Vertical motion value (up/down).
            direction (str): Logical direction used to update opposite_direction context.
        """
        global current_direction, position, opposite_direction
        command = f'rc {left_right} {forward_backward} {up_down} 0'
        opposite_direction = opposites.get(direction, opposite_direction)
        self.send_command(command)

    def recall(self):
        """
        Initiates the recall command to instruct the drone to return to its original position.
        """
        print("Initiating recall from base station...")
        self.send_command("RECALL")

    def land(self):
        """
        Sends the 'land' command to initiate drone landing.
        """
        self.send_command("land")

    def takeoff(self):
        """
        Sends the 'takeoff' command to initiate drone liftoff.
        """
        self.send_command("takeoff")

    def keep_alive(self):
        """
        Sends periodic 'keepalive' messages to maintain connectivity with the drone.
        Executed once every 5 seconds to prevent timeout/disconnection.
        """
        global last_keep_time
        keep_time = time.time()
        if (keep_time - last_keep_time) >= 5:
            self.send_command("keepalive")
            last_keep_time = time.time()


# --- Control Loop ---
def control_drone():
    """
    Main event loop responsible for capturing keyboard inputs and controlling the drone
    accordingly using the pygame interface. It maps directional keys (arrow keys) and
    control commands (T/L/Q/W/S) to drone operations like movement, takeoff, landing, etc.
    """
    global running, position, keyup_flag, last_keep_time
    drone = Drone(stream=True)

    pygame.init()
    pygame.display.set_caption("Drone Control")
    screen = pygame.display.set_mode([960, 720])
    pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)
    running = True
    last_keep_time = time.time()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    drone.update(0, S, 0, "forward")
                    keyup_flag = True
                elif event.key == pygame.K_DOWN:
                    drone.update(0, -S, 0, "backward")
                    keyup_flag = True
                elif event.key == pygame.K_LEFT:
                    drone.update(-S, 0, 0, "left")
                    keyup_flag = True
                elif event.key == pygame.K_RIGHT:
                    drone.update(S, 0, 0, "right")
                    keyup_flag = True
                elif event.key == pygame.K_w:
                    drone.update(0, 0, S, "hover")
                    keyup_flag = True
                elif event.key == pygame.K_s:
                    drone.update(0, 0, -S, "hover")
                    keyup_flag = True
                elif event.key == pygame.K_t:
                    drone.takeoff()
                    keyup_flag = False
                elif event.key == pygame.K_l:
                    drone.land()
                    keyup_flag = False
                elif event.key == pygame.K_q:
                    drone.recall()
                    keyup_flag = False
            elif event.type == pygame.KEYUP:
                if keyup_flag:
                    drone.update(0, 0, 0, "hover")

        drone.keep_alive()
        pygame.display.update()


# --- Main function ---
def main():
    """
    Entry point for the base station control system. Initializes and runs three parallel threads:
    1. receive_data: Collects telemetry and alerts from the Raspberry Pi.
    2. control_drone: Captures keyboard input for manual drone navigation.
    3. receive_file: Handles incoming JSON flight data files for mapping.
    All threads operate asynchronously to ensure responsive and uninterrupted operation.
    """
    global running
    try:
        data_thread = threading.Thread(target=receive_data, daemon=True)
        control_thread = threading.Thread(target=control_drone, daemon=True)
        file_thread = threading.Thread(target=receive_file, daemon=True)

        data_thread.start()
        control_thread.start()
        file_thread.start()

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        running = False
        data_thread.join()
        control_thread.join()
        control_thread.join()


if __name__ == '__main__':
    main()



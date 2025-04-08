#rassberypi
# --- Importing required libraries for hardware control, communication, timing, and logging ---
import socket  # Library for network communication (TCP/UDP)
import smbus2  # Library for I2C communication with sensors
import VL53L1X  # Library for interacting with VL53L1X distance sensors
import RPi.GPIO as GPIO  # Raspberry Pi GPIO pin control
import time  # For timing and delays
import logging  # For system logging
import threading  # For multithreading
import subprocess  # For executing system commands (e.g., I2C address detection)
import datetime  # For timestamping files
import json  # For handling JSON-formatted files

# --- Configure the logging level to INFO ---
logging.basicConfig(level=logging.INFO)

# --- General system variables ---
ID = 0  # Unique drone identifier
my_ip = "192.168.2.11"  # IP address of the Raspberry Pi
PIE_PORT = 12345  # Port for receiving commands from the base station
SERVER_PORT = 65432  # Port for sending text messages to the base station
FILE_PORT = 65436  # Port for sending data files to the base station
TELLO_ADDR = ('192.168.10.1', 8889)  # IP address of the Tello drone

# --- Direction, distance, and system state variables ---
last_direction = "hover"
current_direction = "hover"
last_sample_direction = current_direction
last_last_sample_direction = current_direction
alert_distance = 70  # Safety threshold distance in centimeters
last_distances = {"forward": 220, "backward": 220, "left": 220, "right": 220}
last_last_distances = {"forward": 220, "backward": 220, "left": 220, "right": 220}
sensor_distances = {"forward": 220, "backward": 220, "left": 220, "right": 220}
opposites = {"forward": "backward", "backward": "forward", "left": "right", "right": "left", "hover": "hover"}
opposite_direction = "hover"
is_recalling = False  # Indicates if the drone is currently returning to the starting point
command_stack = []  # Stack of executed commands for recall
DISTANCE_TOLERANCE = 40  # Tolerance threshold for distance comparison
last_command_time = 0
alert = False
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Timestamp for naming files
FILE_NAME = f"flight_data_{timestamp}.json"
start_time = time.time()
run = False
command_hover = "rc 0 0 0 0"  # Command to stop all motion (hover in place)
S = 25  # Default movement speed
last_revaluation_speed = S
count = 0
sample_time = 2
last_sample_time = 1
last_last_sample_time = 0
battery_diff = 0
last_battery = 100  # Initial battery level

# --- GPIO pins used to activate (XSHUT) each distance sensor ---
XSHUT_FORWARD = 6
XSHUT_BACKWARD = 22
XSHUT_LEFT = 26
XSHUT_RIGHT = 24

# --- Unique I2C addresses assigned to each VL53L1X sensor ---
I2C_ADDRESS_FORWARD = 0x30
I2C_ADDRESS_BACKWARD = 0x32
I2C_ADDRESS_RIGHT = 0x34
I2C_ADDRESS_LEFT = 0x36

# --- Sensor initialization function with address assignment ---
def initialize_sensor(xshut_pin, new_address):
    """
    Initializes a VL53L1X sensor by activating its XSHUT pin, opening the I2C bus,
    changing the I2C address to a unique value, and scanning for address confirmation.
    """
    GPIO.output(xshut_pin, GPIO.HIGH)
    time.sleep(1)
    sensor = VL53L1X.VL53L1X(i2c_bus=1)
    sensor.open()
    sensor.change_address(new_address)
    sensor.close()
    sensor.open()
    addresses = subprocess.run(['i2cdetect', '-y', '1'], capture_output=True, text=True)
    print(addresses.stdout)
    return sensor

# --- Sending messages to the base station with retry mechanism ---
def send_message(message, retries=3):
    """
    Sends a textual message to the base station over TCP.
    Implements a retry mechanism to enhance reliability in unstable network conditions.
    """
    for attempt in range(retries):
        try:
            message_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            message_client_socket.connect(('192.168.2.1', SERVER_PORT))
            message_client_socket.sendall(message.encode())
            logging.info(f"Message sent: {message}")
            message_client_socket.close()
            break
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(1)
    else:
        logging.error(f"Failed to send message after {retries} attempts: {message}")

# --- Sending a data file to the base station ---
def send_file_to_basestation(file_name, file_type="ORIGINAL"):
    """
    Sends a JSON-formatted file containing telemetry data to the base station.
    Includes a metadata header with file type and name.
    """
    try:
        file_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        file_client_socket.connect(('192.168.2.1', FILE_PORT))

        # Transmit metadata header padded to 1KB
        header = json.dumps({"type": file_type, "filename": file_name})
        file_client_socket.sendall(header.encode().ljust(1024))

        # Transmit file contents
        with open(file_name, "rb") as file:
            file_data = file.read()
            file_client_socket.sendall(file_data)

        logging.info(f"Sent {file_type} file {file_name} to base station.")
        time.sleep(3)
        ack = file_client_socket.recv(1024).decode()
        if ack == "FILE_RECEIVED":
            logging.info("Base station confirmed file receipt.")
        else:
            logging.warning("No confirmation received from base station.")
        file_client_socket.close()
    except Exception as e:
        logging.error(f"Error sending file to base station: {e}")

# --- Distance sensor system class ---
class SensorSystem:
    """
    This class manages the initialization and operation of multiple VL53L1X distance sensors
    over the I2C interface. It provides synchronized distance measurements from all four directions.
    """
    def __init__(self):
        # Configure GPIO mode and set all XSHUT pins to OUTPUT mode
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(XSHUT_FORWARD, GPIO.OUT)
        GPIO.setup(XSHUT_BACKWARD, GPIO.OUT)
        GPIO.setup(XSHUT_LEFT, GPIO.OUT)
        GPIO.setup(XSHUT_RIGHT, GPIO.OUT)

        # Deactivate all sensors initially to avoid address conflicts
        GPIO.output(XSHUT_FORWARD, GPIO.LOW)
        GPIO.output(XSHUT_BACKWARD, GPIO.LOW)
        GPIO.output(XSHUT_LEFT, GPIO.LOW)
        GPIO.output(XSHUT_RIGHT, GPIO.LOW)
        time.sleep(0.1)

        # Initialize each sensor sequentially with a unique I2C address
        self.sensor_forward = initialize_sensor(XSHUT_FORWARD, I2C_ADDRESS_FORWARD)
        self.sensor_backward = initialize_sensor(XSHUT_BACKWARD, I2C_ADDRESS_BACKWARD)
        self.sensor_left = initialize_sensor(XSHUT_LEFT, I2C_ADDRESS_LEFT)
        self.sensor_right = initialize_sensor(XSHUT_RIGHT, I2C_ADDRESS_RIGHT)

    def get_distances(self):
        """
        Activates all sensors and retrieves current distance measurements in centimeters.
        Returns a dictionary of distances for all four directions.
        In case of sensor failure, defaults to 220 cm.
        """
        self.sensor_forward.start_ranging(1)
        self.sensor_backward.start_ranging(1)
        self.sensor_left.start_ranging(1)
        self.sensor_right.start_ranging(1)

        sensor_distance = {
            "forward": (self.sensor_forward.get_distance() / 10) or 220,
            "backward": (self.sensor_backward.get_distance() / 10) or 220,
            "left": (self.sensor_left.get_distance() / 10) or 220,
            "right": (self.sensor_right.get_distance() / 10) or 220
        }
        logging.info(f"Distances: {sensor_distance} cm")
        return sensor_distance
# --- Battery Monitoring Function ---
def get_battery(control_socket_tello):
    """
    Queries the Tello drone for the current battery level.
    In case of communication failure, an estimated battery value is returned based on the last known difference.
    """
    global last_battery, battery_diff
    try:
        control_socket_tello.sendto(b'battery?', TELLO_ADDR)
        response, _ = control_socket_tello.recvfrom(1024)
        battery_level = int(response.decode())
        battery_diff = last_battery - battery_level
        last_battery = battery_level
        print(f"Battery level: {battery_level}%")
        return battery_level
    except Exception as e:
        print(f"Battery check failed: {e}")
        battery_level = last_battery - battery_diff
        last_battery = battery_level
        return battery_level

# --- Basic Noise Filtering Inspired by Kalman Filter ---
def filter_distance(measured, previous, alpha=0.7, threshold=50):
    """
    Implements a simple distance filtering technique inspired by the Kalman Filter.
    If the difference between the measured and predicted values exceeds the threshold,
    the predicted value is favored. Otherwise, a weighted average is used.
    """
    if abs(measured - previous) > threshold:
        estimated = previous  # Sudden jump detected – trust prediction
    else:
        estimated = alpha * previous + (1 - alpha) * measured  # Weighted average
    return estimated

# --- Converts Movement Direction to Command ---
def direction_to_command(direction1):
    """
    Maps logical direction commands (forward, backward, left, right) into Tello drone remote control syntax.
    """
    if direction1 == "forward":
        change_command = f"rc 0 {S} 0 0"
    elif direction1 == "backward":
        change_command = f"rc 0 {-S} 0 0"
    elif direction1 == "right":
        change_command = f"rc {S} 0 0 0"
    elif direction1 == "left":
        change_command = f"rc {-S} 0 0 0"
    else:
        change_command = command_hover
    return change_command

# --- Logs the Executed Command for Future Recall ---
def save_command(data1):
    """
    Saves the executed drone command along with its elapsed execution time and the distance in the opposite direction.
    This information is used during autonomous return-to-origin (recall) operations.
    """
    global last_command_time
    if last_command_time == 0:
        last_command_time = time.time()
    current_time = time.time()
    elapsed_time = current_time - last_command_time
    if opposite_direction != "hover":
        distance1 = sensor_distances[opposite_direction]
    else:
        distance1 = 100  # Default when no movement
    command_stack.append((data1, elapsed_time, distance1))
    print(f"save command: {data1}, {elapsed_time}, {distance1}")
    last_command_time = current_time

# --- Filter Available Movement Directions (excluding current & opposite) ---
def available_distances_direction(current_direction2, opposite_direction2, distances2):
    """
    Filters out the current and opposite directions from the given distance dictionary,
    providing alternative directions to consider for maneuvering.
    """
    available_distances_direction1 = {
        d: distances2[d]
        for d in distances2
        if d != current_direction2 and d != opposite_direction2
    }
    return available_distances_direction1

# --- Recommends a Safer Direction Based on Available Space ---
def recommend_direction(current_direction1, opposite_direction1, distances1):
    """
    Recommends the best alternative direction for the drone based on the largest available distance.
    If all options are below a certain threshold, the function may recommend the opposite direction.
    """
    print("check recommand direction")
    available_dirs = available_distances_direction(current_direction1, opposite_direction1, distances1)
    recommended = max(available_dirs, key=available_dirs.get)
    if current_direction1 == "hover":
        print(f"the reccomand direction is: {recommended}")
        return recommended
    if available_dirs[recommended] < 80:
        recommended = opposite_direction1
    print(f"the reccomand direction is: {recommended}")
    return recommended

# --- Updates Drone Direction and Sends Command ---
def change_direction(control_socket_tello, recommended_dir):
    """
    Updates the current direction of the drone and issues the corresponding movement command.
    The command is also stored in the command stack for potential recall use.
    """
    global current_direction, opposite_direction, last_direction
    last_direction = current_direction
    logging.info(f"change to recommended direction: {recommended_dir}")
    change_command = direction_to_command(recommended_dir)
    current_direction = recommended_dir
    opposite_direction = opposites.get(current_direction, opposite_direction)
    save_command(change_command)
    control_socket_tello.sendto(change_command.encode('utf-8'), TELLO_ADDR)
    return current_direction

# --- Handles Cases with No Clear Path (e.g., Dead Ends) ---
def no_way_out(control_socket_tello):
    """
    In the event of a blocked path with no obvious way forward, this function
    attempts to redirect the drone toward the most spacious available direction.
    """
    global current_direction, opposite_direction
    available_dirs = available_distances_direction(current_direction, opposite_direction, sensor_distances)
    recommended3 = max(available_dirs, key=available_dirs.get)
    if available_dirs[recommended3] > 100:
        current_direction = change_direction(control_socket_tello, recommended3)
    return current_direction

# --- Obstacle Detection and Response ---
def check_alert(control_socket_tello):
    """
    Continuously monitors the drone's current movement direction for potential obstacles.
    If a sudden change in measured distance is detected or if the sensor reports a distance
    below the alert threshold, the drone halts and reroutes using the most suitable direction.
    This function incorporates prediction and filtering logic for noise reduction.
    """
    global current_direction, opposite_direction, sensor_distances, last_command_time, count, last_revaluation_speed
    if current_direction != "hover":
        if last_sample_direction == last_last_sample_direction == current_direction:
            revaluation_speed = ((last_last_distances[current_direction] - last_distances[current_direction]) /
                                 (last_sample_time - last_last_sample_time))
            last_revaluation_speed = revaluation_speed
        else:
            revaluation_speed = last_revaluation_speed

        if 0 < sensor_distances[current_direction] < alert_distance:
            difference_time = sample_time - last_sample_time
            limit_distance = difference_time * revaluation_speed * 2
            if abs(sensor_distances[current_direction] - last_distances[current_direction]) < limit_distance or count >= 2:
                save_command(command_hover)
                control_socket_tello.sendto(command_hover.encode('utf-8'), TELLO_ADDR)
                distances_average = {
                    d: ((sensor_distances[d] + last_distances[d]) / 2) for d in sensor_distances
                }
                recommended2 = recommend_direction(current_direction, opposite_direction, distances_average)
                current_direction = change_direction(control_socket_tello, recommended2)
                send_message(f"ALERT: Obstacle detected in {last_direction} at {sensor_distances[current_direction]:.2f} cm, "
                             f"change to recommended direction: {current_direction}")
                count = 0
                return True

            logging.warning(f"Possible sensor glitch in {current_direction}: "
                            f"{last_distances[current_direction]} -> {sensor_distances[current_direction]}, "
                            f"time delta: {difference_time}, count: {count}")
            predicted = last_distances[current_direction] - (revaluation_speed * difference_time)
            measured = sensor_distances[current_direction]
            filtered = filter_distance(measured, predicted, alpha=0.7, threshold=30)
            sensor_distances[current_direction] = max(filtered, 0.1)
            count += 1
        else:
            count = 0
    elif current_direction == "hover":
        count = 0
        for direction, distance in sensor_distances.items():
            if 0 < distance < alert_distance:
                recommended_dir = recommend_direction(current_direction, opposite_direction, sensor_distances)
                send_message(f"ALERT: Obstacle detected in {direction} at {distance:.2f} cm, "
                             f"recommended direction: {recommended_dir}")
                opposite_direction = opposites.get(current_direction, opposite_direction)
                command1 = direction_to_command(opposite_direction)
                control_socket_tello.sendto(command1.encode('utf-8'), TELLO_ADDR)
                time.sleep(1)
                control_socket_tello.sendto(command_hover.encode('utf-8'), TELLO_ADDR)
                return True
    return False

# --- Save Real-Time Data to JSON File ---
def save_to_file(distances1, current_direction1):
    """
    Logs the current distance measurements and drone direction into a JSON file for post-flight analysis.
    """
    elapsed_time1 = round(time.time() - start_time, 2)
    record = {
        "time": elapsed_time1,
        "current_direction": current_direction1,
        "distance": distances1,
    },
    with open(FILE_NAME, "a") as file:
        file.write(json.dumps(record, indent=3) + "\n")

# --- Save Data Specifically During Recall Phase ---
def save_to_recall_file(distances1, current_direction1):
    """
    Stores sensor data and direction during the autonomous return-to-origin phase.
    Used to evaluate the accuracy and consistency of the recall algorithm.
    """
    elapsed_time1 = round(time.time() - start_time, 2)
    record = {
        "time": elapsed_time1,
        "current_direction": current_direction1,
        "distance": distances1,
    },
    with open(f"recall_{FILE_NAME}", "a") as file:
        file.write(json.dumps(record, indent=3) + "\n")

# --- Monitor Sensor Data and Control Drone Behavior ---
def monitor_distances(control_socket_tello, sensor_system):
    """
    Continuously monitors real-time distance data from all onboard sensors.
    This function is responsible for logging measurements, detecting obstacles,
    verifying path availability, evaluating battery level, and managing recall transitions.
    It operates as a background thread and updates global direction states.
    """
    global sensor_distances, is_recalling, current_direction, last_distances, sample_time, last_sample_time
    global last_battery, last_last_distances, last_last_sample_time, last_sample_direction, last_last_sample_direction

    last_opposite_direction = "hover"
    sensor_distances = sensor_system.get_distances()
    sample_time = time.time()
    count_direction = 0

    # Initial battery level measurement
    control_socket_tello.sendto(b'battery?', TELLO_ADDR)
    response, _ = control_socket_tello.recvfrom(1024)
    last_battery = int(response.decode())
    battery_start = last_battery

    while True:
        try:
            # Update time and distance buffers
            last_last_distances = last_distances
            last_distances = sensor_distances
            last_last_sample_time = last_sample_time
            last_sample_time = sample_time

            # Read new sensor values
            sensor_distances = sensor_system.get_distances()
            sample_time = time.time()

            if run:
                if is_recalling:
                    save_to_recall_file(sensor_distances, current_direction)
                else:
                    battery_level = get_battery(control_socket_tello)

                    # Only operate if battery is above 60% of starting value
                    if battery_level >= battery_start * 0.6:
                        save_to_file(sensor_distances, current_direction)

                        # Handle potential deadlock by forcing change of direction
                        while current_direction == last_opposite_direction:
                            last_last_distances = last_distances
                            last_distances = sensor_distances
                            last_sample_time = sample_time
                            sensor_distances = sensor_system.get_distances()
                            sample_time = time.time()
                            last_last_sample_direction = last_sample_direction
                            last_sample_direction = current_direction
                            current_direction = no_way_out(control_socket_tello)

                        # Send current status update
                        send_message(
                            f"DATA: Direction={current_direction}, Forward={sensor_distances['forward']:.2f}, "
                            f"Backward={sensor_distances['backward']:.2f}, Left={sensor_distances['left']:.2f}, "
                            f"Right={sensor_distances['right']:.2f}"
                        )

                        last_opposite_direction = opposite_direction

                        # Check for obstacle alert
                        check_alert(control_socket_tello)

                        # Update direction history
                        if current_direction == last_direction:
                            count_direction += 1
                            last_last_sample_direction = last_sample_direction
                            last_sample_direction = current_direction
                        else:
                            last_last_sample_direction = last_sample_direction
                            last_sample_direction = last_direction
                            count_direction = 0

                        # Re-evaluate direction after repeated movement in the same direction
                        if count_direction >= 5:
                            recom = recommend_direction(current_direction, opposite_direction, sensor_distances)
                            if recom != opposite_direction and sensor_distances[recom] > sensor_distances[current_direction]:
                                current_direction = change_direction(control_socket_tello, recom)
                    else:
                        # Initiate return-to-home due to low battery
                        is_recalling = True

                time.sleep(0.0001)  # Minimal sleep for near real-time responsiveness
        except Exception as e:
            logging.error(f"Error in monitor_distances: {e}")
            GPIO.cleanup()


# --- Parse Direction from RC Command String ---
def check_direction(data):
    """
    Parses a command in the format 'rc <lr> <fb> <ud> <yaw>' to determine the corresponding direction.
    Returns one of the values: 'forward', 'backward', 'left', 'right', or 'hover'.
    """
    _, left_right, forward_backward, _, _ = data.split()
    left_right = int(left_right)
    forward_backward = int(forward_backward)

    if forward_backward > 0:
        return "forward"
    elif forward_backward < 0:
        return "backward"
    elif left_right > 0:
        return "right"
    elif left_right < 0:
        return "left"
    return "hover"


# --- Autonomous Return-to-Origin Functionality ---
def recall(control_socket_tello):
    """
    Executes the autonomous recall procedure by reversing all previously issued commands.
    This allows the drone to retrace its path back to the starting point, using timing and distance measurements.
    """
    global current_direction, is_recalling
    print("Returning to start position...")

    if not command_stack:
        print("No commands to recall!")
        send_message("END_RECALL")
        return

    send_message(f"START RECALL:{FILE_NAME}")
    last_elapsed_time = 0

    while command_stack:
        command, elapsed_time, distance1 = command_stack.pop()
        print(f"Commands remaining in stack: {len(command_stack)}")

        command_parts = command.split()
        if command_parts[0] == "rc":
            # Invert command direction
            left_right = -int(command_parts[1])
            forward_backward = -int(command_parts[2])
            up_down = -int(command_parts[3])
            reverse_command = f'rc {left_right} {forward_backward} {up_down} 0'

            current_direction = check_direction(reverse_command)
            print(reverse_command)

            if current_direction != "hover":
                send_message(f"RETURN: Distance={sensor_distances[current_direction]:.2f}")
                control_socket_tello.sendto(reverse_command.encode('utf-8'), TELLO_ADDR)
                print(f"at {time.time()} moving to: {current_direction} for {last_elapsed_time} seconds")
                time.sleep(last_elapsed_time)

                current_distance = sensor_distances[current_direction]
                distance_dif = distance1 - current_distance
                print(f"Distance deviation: {distance_dif}")

                send_message(f"RETURN: Distance={sensor_distances[current_direction]:.2f}")
                count_fix = 0

                while abs(distance_dif) > DISTANCE_TOLERANCE:
                    print(f"distance diff is: {distance_dif}")
                    if distance_dif > 0:
                        control_socket_tello.sendto(reverse_command.encode('utf-8'), TELLO_ADDR)
                    else:
                        control_socket_tello.sendto(command.encode('utf-8'), TELLO_ADDR)

                    time.sleep(1)
                    last_distance = current_distance
                    current_distance = sensor_distances[current_direction]
                    print(f"current distance: {current_distance}")

                    if abs(current_distance - last_distance) <= 10 or count_fix <= 3:
                        distance_dif = 0.1
                        count_fix = 0
                        print("No significant change in distance")
                    else:
                        distance_dif = distance1 - current_distance
                        count_fix += 1

            last_elapsed_time = elapsed_time

    # Final landing and data transfer
    command_stack.clear()
    control_socket_tello.sendto("land".encode('utf-8'), TELLO_ADDR)
    print("Drone has landed.")
    print(f"Sending files to base station: {FILE_NAME}")
    send_file_to_basestation(FILE_NAME, file_type="ORIGINAL")
    send_file_to_basestation(f"recall_{FILE_NAME}", file_type="RECALL")
    logging.info("Recall file transmission complete.")
    is_recalling = False
    send_message("END_RECALL")


# --- Program Entry Point ---
def main():
    """
    The main entry point of the Raspberry Pi drone controller.
    Initializes all sockets and threads, and processes commands from the base station in a loop.
    Handles control messages such as takeoff, movement, recall, and emergency stop.
    The drone can operate autonomously in case of communication loss or upon mission initiation.
    """
    global current_direction, opposite_direction, is_recalling, last_command_time
    global alert, run, last_sample_direction, last_last_sample_direction

    sensor_system = SensorSystem()

    # TCP socket for communication with the base station
    welcom_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    welcom_sock.bind((my_ip, PIE_PORT))
    welcom_sock.listen(1)
    control_socket_bs, addr = welcom_sock.accept()

    # UDP socket for communication with the Tello drone
    control_socket_tello = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Launch background thread responsible for autonomous flight logic and obstacle avoidance.
    # This thread executes real-time monitoring, environmental analysis, and decision making.
    threading.Thread(
        target=monitor_distances,
        args=(control_socket_tello, sensor_system),
        daemon=True
    ).start()

    count_keep = 0

    while True:
        try:
            buff1 = control_socket_bs.recv(20)
            if not buff1:
                logging.info(
                    "No communication with the base station – data is stored locally, drone continues autonomously.")
                if count_keep == 20:
                    is_recalling = True
                if count_keep == 60:
                    break
                time.sleep(5)
                count_keep += 1
                continue

            count_keep = 0
            data = buff1.decode('utf-8')
            print(f"Received command: {data}")

            if data == "end":
                print("End command received.")
                control_socket_tello.sendto("land".encode('utf-8'), TELLO_ADDR)
                control_socket_tello.sendto("streamoff".encode('utf-8'), TELLO_ADDR)
                is_recalling = False
                break

            elif data == "RECALL":
                save_command(command_hover)
                is_recalling = True
                control_socket_bs.send("OK".encode('utf-8'))
                time.sleep(3)
                recall(control_socket_tello)

            elif data.startswith("rc"):
                current_direction = check_direction(data)
                if current_direction != "hover":
                    opposite_direction = opposites.get(current_direction, opposite_direction)
                    if check_alert(control_socket_tello):
                        alert = True
                        control_socket_bs.send("STOPPED".encode('utf-8'))
                else:
                    alert = False

                if not alert:
                    save_command(data)
                    control_socket_tello.sendto(data.encode('utf-8'), TELLO_ADDR)
                    control_socket_bs.send("OK".encode('utf-8'))

            elif data == "keepalive":
                control_socket_bs.send("OK".encode('utf-8'))
                logging.info("KEEP ALIVE")

            else:
                control_socket_tello.sendto(buff1, TELLO_ADDR)
                control_socket_bs.send("OK".encode('utf-8'))

                if data == "takeoff":
                    save_command(command_hover)
                    print("Takeoff command received.")
                    time.sleep(5)

                    # --- AUTONOMOUS FLIGHT ACTIVATION ---
                    # This marks the transition from manual control to autonomous navigation.
                    # The monitor_distances thread becomes active and begins handling direction logic,
                    # including obstacle avoidance, data logging, and dynamic route adaptation.
                    recommended0 = "forward"
                    current_direction = change_direction(control_socket_tello, recommended0)
                    last_sample_direction = current_direction
                    last_last_sample_direction = last_sample_direction
                    time.sleep(2)
                    run = True  # Autonomous flight is now enabled.

        except Exception as e:
            logging.error(f"Error in main: {e}")
            time.sleep(3)
            continue

    control_socket_bs.close()
    control_socket_tello.close()
    print("Raspberry Pi controller has shut down.")

if __name__ == "__main__":
    main()




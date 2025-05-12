import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from rclpy.qos import (
    qos_profile_sensor_data,
)  # Quality of Service settings for real-time data
import time
from ros2_numpy import pose_to_np, to_ackermann
from collections import deque
import numpy as np

from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory, get_package_prefix

class PIDcontroller(Node):
    def __init__(self, model_path: str):
        super().__init__("pid_controller")

        # Define Quality of Service (QoS) for communication
        qos_profile = qos_profile_sensor_data  # Suitable for sensor data
        qos_profile.depth = 1  # Keep only the latest message

        # Create a publisher for sending AckermannDriveStamped messages to the '/autonomous/ackermann_cmd' topic
        self.publisher = self.create_publisher(
            AckermannDriveStamped, "/autonomous/ackermann_cmd", qos_profile
        )

        # Create a subscription to listen for PoseStamped messages from the '/waypoint' topic
        # When a message is received, the 'self.waypoint_callback' function is called
        self.center_subscriber = self.create_subscription(
            PoseStamped, "/waypoint", self.center_waypoint_callback, qos_profile
        )

        self.object_subscriber = self.create_subscription(
            PoseStamped, "/object", self.object_callback, qos_profile
        )

        # Load parameters
        self.params_set = False
        self.declare_params()
        self.load_params()

        # Create a timer that calls self.load_params every 10 seconds (10.0 seconds)
        self.timer = self.create_timer(10.0, self.load_params)

        self.last_error = 0.0
        self.last_time = time.time()
        self.last_steering_angle = 0.0

        # Initialize deque with a fixed length of self.max_out
        # This could be useful to allow the vehicle to temporarily lose the track for up to max_out frames before deciding to stop. (Currently not used yet.)
        self.max_out = 9
        self.success = deque([True] * self.max_out, maxlen=self.max_out)

        self.len_history = 10

        # Load the custom trained YOLO model
        model = self.load_model(model_path)

        # Map class IDs to labels and labels to IDs
        self.id2label = self.model.names
        targets = ["stop", "speed_3mph", "speed_2mph"]
        self.id2target = {
            id: lbl for id, lbl in self.id2label.items() if lbl in targets
        }

        # Variables for handling sign logic
        self.stopped = False
        self.waited = False
        self.sign_dist = 0.0
        self.start_of_stop_time = 0.0
        self.current_speed = 0.0
        self.last_target_speed = 0.0
        self.speed_switch_time = 0.0

        # Newer parameters
        self.closest_speed_sign_dist = float("inf")


    def load_model(self, filepath):
        model = YOLO(filepath)

        self.imgsz = model.args[
            "imgsz"
        ]  # Get the image size (imgsz) the loaded model was trained on.

        # Init model
        print("Initializing the model with a dummy input...")
        im = np.zeros((self.imgsz, self.imgsz, 3))  # dummy image
        _ = model.predict(im)
        print("Model initialization complete.")

        return model

    def calc_speed(self) -> float:
        """
        Slowly ramp up to the desired speed 
        """
        w = self.target_speed
        v = self.last_target_speed
        r = self.ramp_constant
        t = time.time() - self.speed_switch_time

        return (v - w) * (1 - np.exp(-r * t).item()) + v

    
    def object_callback(self, msg: PoseStamped):
        distance = msg.pose.position.x
        sign_type = self.id2target[int(msg.pose.position.z)]

        # If the distance is < 1 meter, update the vehicle speed
        if distance <= self.sign_distance:
            if sign_type == 'stop' and not self.stopped:
                # Always wait at a stop sign
                if not self.waited:
                    self.stopped = True
                    self.start_of_stop_time = time.time()
                else:
                    ...

            elif distance < self.closest_speed_sign_dist:
                # Only look at the closest speed sign
                if sign_type == 'speed_2mph':
                    if self.target_speed != self.speed_2mph:
                        self.last_target_speed = self.target_speed
                        self.speed_switch_time = time.time()
                    self.target_speed = self.speed_2mph
                elif sign_type == 'speed_3mph':
                    if self.target_speed != self.speed_2mph:
                        self.last_target_speed = self.target_speed
                        self.speed_switch_time = time.time()

                    self.target_speed = self.speed_3mph


    def waypoint_callback(self, msg: PoseStamped):
        # Convert incoming pose message to position, heading, and timestamp
        point, heading, timestamp_unix = pose_to_np(msg)

        # If the detected point contains NaN (tracking lost) stop the vehicle
        if np.isnan(point).any():
            ackermann_msg = to_ackermann(0.0, self.last_steering_angle, timestamp_unix)
            self.publisher.publish(ackermann_msg)  # BRAKE
            self.success.append(False)
            self.last_target_speed = 0.0
            self.speed_switch_time = time.time()
            return
        else:
            self.success.append(True)

        if time.time() - self.start_of_stop_time < self.stop_time:
            # Handle stop sign
            ackermann_msg = to_ackermann(0.0, 0.0, timestamp)
            self.publisher.publish(ackermann_msg)
            self.last_target_speed = 0.0
            self.speed_switch_time = time.time()
            return
        else:
            self.stopped = False
            self.waited = True
         
        # Calculate time difference since last callback
        dt = timestamp_unix - self.last_time

        # Update the last time to the current timestamp
        self.last_time = timestamp_unix

        # Get x and y coordinates (ignore z), and compute the error in y
        x, y, _ = point
        error = 0.0 - y  # Assuming the goal is to stay centered at y = 0

        # Calculate the derivative of the error (change in error over time)
        d_error = (error - self.last_error) / dt

        # Compute the steering angle using a PD controller
        steering_angle = (self.kp * error) + (self.kd * d_error)

        # Get the timestamp from the message header
        timestamp = msg.header.stamp

        # Create an Ackermann drive message with speed and steering angle
        ackermann_msg = to_ackermann(self.calc_speed(), steering_angle, timestamp)

        # Publish the message to the vehicle
        self.publisher.publish(ackermann_msg)

        # Save the current error for use in the next iteration
        self.last_error = error

        # If tracking is lost, continue driving with the last known steering angle
        # to follow the previously estimated path (e.g., maintain the current curve)
        self.last_steering_angle = steering_angle

    def stop_callback(self, msg: PoseStamped):
        point, heading, timestamp_unix = pose_to_np(msg)

        distance = msg.pose.position.x

        # Handle double stop sign
        if self.sign_dist < distance:
            self.stopped = False

        self.sign_dist = distance

        if not np.isnan(point).any():
            # If the distance is < 1 meter, update the vehicle speed
            if distance <= self.sign_distance:
                self.get_logger().info(f"Stop sign detected at distance {distance:.2f}m. Wait 2 seconds and go.")
                
                if not self.stopped:
                    self.start_of_stop_time = time.time()
                    self.stopped = True


    def speed_2mph_waypoint_callback(self, msg: PoseStamped):
        point, heading, timestamp_unix = pose_to_np(msg)

        self.stopped = False
        distance = msg.pose.position.x
        if not np.isnan(point).any():
            # If the distance is < 1 meter, update the vehicle speed
            if distance <= self.sign_distance:
                self.get_logger().info(
                    f"2mph sign detected at distance {distance:.2f}m."
                )

                if self.target_speed != self.speed_2mph:
                    self.last_target_speed = self.target_speed
                    self.speed_switch_time = time.time()
                self.target_speed = self.speed_2mph  # 2 mph in m/s

    def speed_3mph_waypoint_callback(self, msg: PoseStamped):
        point, heading, timestamp_unix = pose_to_np(msg)

        self.stopped = False
        distance = msg.pose.position.x
        if not np.isnan(point).any():
            # If the distance is < 1 meter, update the vehicle speed
            if distance <= self.sign_distance:
                self.get_logger().info(
                    f"3mph sign detected at distance {distance:.2f}m, increasing speed."
                )

                if self.target_speed != self.speed_2mph:
                    self.last_target_speed = self.target_speed
                    self.speed_switch_time = time.time()
                self.target_speed = self.speed_3mph  # 3 mph in m/s

    def declare_params(self):

        # Declare parameters with default values
        self.declare_parameters(
            namespace="",
            parameters=[
                ("kp", 0.9),
                ("kd", 0.0),
                ("speed_2mph", 0.89408),  # [m/s]
                ("speed_3mph", 1.34112),  # [m/s]
                ("sign_dist", 1.0),  # [m]
                ("stop_time", 2.0),
                ("ramp_constant", 2.0),
            ],
        )

    def load_params(self):
        try:
            self.kp = self.get_parameter("kp").get_parameter_value().double_value
            self.kd = self.get_parameter("kd").get_parameter_value().double_value
            self.speed_2mph = self.get_parameter("speed_2mph").get_parameter_value().double_value
            self.speed_3mph = self.get_parameter("speed_3mph").get_parameter_value().double_value
            self.sign_dist = self.get_parameter("sign_dist").get_parameter_value().double_value
            self.stop_time = self.get_parameter("stop_time").get_parameter_value().double_value
            self.ramp_constant = self.get_parameter("ramp_constant").get_parameter_value().double_value
            self.speed_2mph = (
                self.get_parameter("speed_2mph").get_parameter_value().double_value
            )
            self.speed_3mph = (
                self.get_parameter("speed_3mph").get_parameter_value().double_value
            )
            self.sign_dist = (
                self.get_parameter("sign_dist").get_parameter_value().double_value
            )
            self.stop_time = (
                self.get_parameter("stop_time").get_parameter_value().double_value
            )
            self.ramp_constant = (
                self.get_parameter("ramp_constant").get_parameter_value().double_value
            )
            self.target_speed = self.speed_2mph

            if not self.params_set:
                self.get_logger().info("Parameters loaded successfully")
                self.params_set = True

        except Exception as e:
            self.get_logger().error(f"Failed to load parameters: {e}")


def main(args=None):
    rclpy.init(args=args)

    # Path to your custom trained YOLO model
    pkg_path = get_package_prefix("line_follower").replace("install", "src")
    model_path = pkg_path + "/models/best.pt"

    node = PIDcontroller(model_path=model_path)
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

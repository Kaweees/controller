# Third-Party Libraries
import os
import time
from collections import deque
import numpy as np
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory, get_package_prefix

# ROS Imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from rclpy.qos import qos_profile_sensor_data  # Quality of Service settings for real-time data

# Project-Specific Imports
from ros2_numpy import pose_to_np, to_ackermann


class PIDcontroller(Node):
    def __init__(self, model_path: str):
        '''
        Initializes a pid controller ROS node. Listens to object info to generate control commands.
        Subscribes to:
            /waypoint   = center of the line to follow in (real world coordinates).
            /object     = object id and position (real world coordinates).
        Publishes:
            /autonomous/ackermann_cmd   = speed and steering commands (ackermann) for autonomous driving.
        '''
        super().__init__('pid_controller')

        # Ensure model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The file at '{model_path}' was not found.")

        # Define Quality of Service (QoS) for communication
        qos_profile = qos_profile_sensor_data  # Suitable for sensor data
        qos_profile.depth = 1  # Keep only the latest message

        # Create a subscription to listen for PoseStamped messages from the '/waypoint' topic
        # When a message is received, the 'self.waypoint_callback' function is called
        self.way_sub = self.create_subscription(PoseStamped,'/waypoint', self.waypoint_callback, qos_profile)
        # Create a subscription to listen for PoseStamped messages from the '/object' topic
        self.obj_sub = self.create_subscription(PoseStamped,'/object', self.obj_callback, qos_profile)

        # Create a publisher for sending AckermannDriveStamped messages to the '/autonomous/ackermann_cmd' topic
        self.ack_cmd_publisher = self.create_publisher(AckermannDriveStamped, '/autonomous/ackermann_cmd', qos_profile)

        # Load parameters
        self.params_set = False
        self.declare_params()
        self.load_params()

        # Create a timer that calls self.load_params every 10 seconds (10.0 seconds)
        self.timer = self.create_timer(10.0, self.load_params)

        # Initialize 'previous' values for the lost line case
        self.last_error = 0.0
        self.last_time = time.time()
        self.last_steering_angle = 0.0

        # Initialize deque with a fixed length of self.max_not_found
        # This could be useful to allow the vehicle to temporarily lose the track for up to max_out frames before deciding to stop.
        # (Currently not used yet.)
        self.max_not_found = 9
        self.line_found = deque([True] * self.max_not_found, maxlen=self.max_not_found)

        # Load the custom trained YOLO model
        model = YOLO(model_path)
        # Map class IDs to labels and labels to IDs
        id2label = model.names
        targets = ['stop', 'speed_3mph', 'speed_2mph']
        self.center_line_id: list[int] = [id_ for id_, lbl in self.id2label.items() if lbl == 'center']
        self.id2target = {id: lbl for id, lbl in id2label.items() if lbl in targets}

        # Length of history for objects detected
        self.len_history = 10
        # Initialize target metadata for each label (e.g. stop sign, speed signs)
        self.targets = {
            lbl: {
                'id': id,  # Numerical class ID
                'history': deque([False] * self.len_history, maxlen=self.len_history),  # Detection history
                'detected': False,   # Whether the target is currently detected
                'reacted': False,    # Whether the vehicle has already reacted
                'threshold': 0.5,    # Detection threshold (percent of history with positive detection)
                'distance': 0.0,     # Current distance to the target
                'min_distance': 2.0  # Distance threshold to start reacting
            }
            for id, lbl in self.id2target.items()
        }

        # Log an informational message indicating that the PID Controller Node has started
        self.get_logger().info("PID Controller Node started.")
        self.get_logger().info(f"Parsing predictions for {self.id2target} and {self.center_line_id}:center")


    def waypoint_callback(self, msg: PoseStamped):
        '''
        Callback for each /waypoint pose message received via subscription.
        Converts waypoint into steering angle and publishes autonomous ackermann commands based on this information.
        '''
        # Convert incoming pose message to position, heading, and timestamp
        point, heading, timestamp_unix = pose_to_np(msg)

        # If the detected point contains NaN (tracking lost line) stop the vehicle 
        if np.isnan(point).any():
            # Indicate that the line was not detected in this frame 
            self.line_found.append(False)
            # Check if the line was detected within the previous self.max_not_found frames
            if any(self.line_found):
                # Keep driving with the last known steering angle unless the line is lost for self.max_not_found consecutive frames
                ackermann_msg = to_ackermann(self.speed, self.last_steering_angle, timestamp_unix)
            else:
                # Line hasn't been detected in self.max_not_found frames, stop the vehicle
                ackermann_msg = to_ackermann(0.0, self.last_steering_angle, timestamp_unix)
                self.ack_cmd_publisher.publish(ackermann_msg) # BRAKE
                
                self.get_logger().info(f"Line not found in {self.max_not_found} frames")
            # Don't update the 'previous' values
            return
        
        # Line detected
        else:
            # Indicate that the line was detected in this frame
            self.line_found.append(True)

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
            ackermann_msg = to_ackermann(self.speed, steering_angle, timestamp)
            # Publish the message to the vehicle
            self.ack_cmd_publisher.publish(ackermann_msg)

            # Save the current error for use in the next iteration
            self.last_error = error

            # Save the current steering angle for use in the next iteration
            # If tracking is lost, continue driving with the last known steering angle
            # to follow the previously estimated path (e.g., maintain the current curve)
            self.last_steering_angle = steering_angle


    def obj_callback(self, msg: PoseStamped):
        '''
        Callback for each /object pose message received via subscription.
        Handles each of the important objects detected.
        '''
        # Convert incoming pose message to position, heading, and timestamp
        point, heading, timestamp_unix = pose_to_np(msg)

        # Extract object information from message
        id = int(point[-1]) # Extract class ID stored in the z-coordinate
        x, _ = point[:2] # Get distance
        if id not in self.id2target:
            # Not an important object
            self.get_logger().info(f"Detected object with id:{id}, not dealing with it")
            return
        lbl = self.id2target[id]
        data = self.targets[lbl]

        # Check if point is NaN (object not detected or not visible)
        if np.isnan(point).any():
            # Indicate object not found in the current frame
            data['history'].append(False)
        else:
            # Indicate object found in the current frame 
            data['history'].append(True)
            data['distance'] = x

        # Handle each important object
        if lbl == 'stop':
            self.stop_and_wait(data) # stop sign
        elif lbl == 'speed_2mph':
            self.change_speed(data, 2) # 2 mph speed limit sign
        elif lbl == 'speed_3mph':
            self.change_speed(data, 3) # 3 mph speed limit sign


    def stop_and_wait(self, target_data, duration=2.0):
        '''
        Handles detection of a stop sign object.
        '''
        # Update detection flag based on average detection history
        target_data['detected'] = True if np.mean(target_data['history']) > target_data['threshold'] else False

        # Only stop if target_data is detected, not yet reacted, and close enough
        if target_data['detected'] and not target_data['reacted']:
            if target_data['distance'] < target_data['min_distance']:
                # Create message to stop car
                ackermann_msg = to_ackermann(0.0, self.last_steering_angle)
                # Publish stop message
                self.ack_cmd_publisher.publish(ackermann_msg)  # BRAKE
                # Sleep to wait at the stop sign for the specified duration
                time.sleep(duration)
                # Mark target as handled i.e. already reacted to it
                target_data['reacted'] = True
                # Indicate moving past the stop sign
                self.get_logger().info("Done with sleep, now moving past the stop sign")

        # Check if the target_data hasn't been detected in any recent frames
        if not any(target_data['history']) and target_data['reacted']:
            # Reset reaction flag
            target_data['reacted'] = False

    
    def change_speed(self, target_data, speed_limit):
        '''
        Handles detection of a speed limit sign.
        '''
        # Update detection flag based on average detection history 
        target_data['detected'] = True if np.mean(target_data['history']) > target_data['threshold'] else False

        # Only change speed if target_data is detected, not yet reacted, and close enough
        if target_data['detected'] and not target_data['reacted']:
            if target_data['distance'] < target_data['min_distance']:
                # Update speed
                self.speed = self._mph_to_mps(speed_limit)
                # Mark target as handled i.e. already reacted to it
                target_data['reacted'] = True
                # Indicate changing speed
                self.get_logger().info(f"Changing speed for {speed_limit} mph speed limit sign")

        # Check if the target_data hasn't been detected in any recent frames
        if not any(target_data['history']) and target_data['reacted']:
            # Reset reaction flag
            target_data['reacted'] = False


    def _mph_to_mps(self, mph):
        '''
        Converts miles per hour to meters per second.
        '''
        # return mph * 5280 / (60 * 60 * 3.28)
        # return mph * 0.44715447154471544
        return mph * 0.44704
    

    def declare_params(self):
        '''
        Declares parameters. idk why.
        '''
        # Declare parameters with default values
        self.declare_parameters(
            namespace='',
            parameters=[
                ('kp', 0.9),
                ('kd', 0.0),
                ('speed', 0.6),
            ]
        )


    def load_params(self):
        '''
        Loads parameters from config file ?
        '''
        try:
            self.kp = self.get_parameter('kp').get_parameter_value().double_value
            self.kd = self.get_parameter('kd').get_parameter_value().double_value
            self.speed = self.get_parameter('speed').get_parameter_value().double_value

            if not self.params_set:
                self.get_logger().info("Parameters loaded successfully")
                self.params_set = True

        except Exception as e:
            self.get_logger().error(f"Failed to load parameters: {e}")


def main(args=None):
    # Path to your custom trained YOLO model
    # /mxck2_ws/install/line_follower → /mxck2_ws/src/line_follower
    pkg_path = get_package_prefix('line_follower').replace('install', 'src') 
    model_path = pkg_path + '/models/best.pt'

    rclpy.init(args=args)
    node = PIDcontroller(model_path)
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

import launch
import launch_ros.actions

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Path to the configuration file for the line follower node
    config_file_path = get_package_share_directory('controller') + '/config/pid_gains.yaml'
    
    # PID node declaration from the controller package
    pid_node = launch_ros.actions.Node(
        package='controller',
        executable='pid_node',
        name='pid_node',
        output='screen',
        parameters=[config_file_path]
    )

    # Return launch description including both nodes
    return launch.LaunchDescription([
        pid_node
    ])

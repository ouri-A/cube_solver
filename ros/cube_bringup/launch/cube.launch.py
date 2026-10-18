"""
Start the whole cube pipeline: scanner -> executor -> viewer.

    ros2 launch cube_bringup cube.launch.py

Every node reads its parameters from config/cube.yaml in this package.
"""

from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params = PathJoinSubstitution([FindPackageShare('cube_bringup'), 'config', 'cube.yaml'])
    return LaunchDescription([
        Node(package='cube_scanner', executable='scanner', name='scanner',
             parameters=[params], output='screen'),
        Node(package='cube_executor', executable='executor', name='executor',
             parameters=[params], output='screen'),
        Node(package='cube_viewer', executable='viewer', name='viewer',
             parameters=[params], output='screen'),
    ])

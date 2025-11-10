from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Camera
        Node(
            package='depthai_ros_driver',
            executable='camera_node',
            name='oak_camera'
        ),
        
        # Learning collector
        Node(
            package='dexhand_ai',
            executable='learning_collector',
            name='collector'
        ),
        
        # Gesture controller (sim or hardware)
        Node(
            package='dexhand_gesture_controller',
            executable='gesture_controller',
            name='gesture_controller',
            parameters=[{'simulation': LaunchConfiguration('sim', default=True)}]
        )
    ])
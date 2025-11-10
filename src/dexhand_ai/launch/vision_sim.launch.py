from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_topic',
            default_value='/camera/color/image_raw',
            description='Camera topic'
        ),
        
        # Camera node (OAK-1 or webcam)
        Node(
            package='depthai_ros_driver',
            executable='camera_node',
            name='oak_camera',
            parameters=[{
                'camera_model': 'OAK-1',
                'fps': 30,
                'resolution': '1080p'
            }],
            condition=LaunchConfiguration('use_oak', default='true')
        ),
        
        # Webcam fallback
        Node(
            package='image_tools',
            executable='cam2image',
            name='webcam',
            parameters=[{
                'width': 640,
                'height': 480,
                'fps': 30
            }],
            condition=LaunchConfiguration('use_webcam', default='false')
        ),
        
        # AI Gesture Bridge
        Node(
            package='dexhand_ai',
            executable='ai_gesture_bridge',
            name='ai_bridge',
            parameters=[os.path.join('config', 'object_gesture_map.yaml')]
        ),
        
        # DexHand Simulation
        Node(
            package='dexhand_gesture_controller',
            executable='gesture_controller',
            name='gesture_controller',
            parameters=[{'simulation': True}]
        ),
        
        # RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', os.path.join('config', 'dexhand_view.rviz')]
        )
    ])
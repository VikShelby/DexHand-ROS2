from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare the launch arguments
    declared_arguments = [
        DeclareLaunchArgument(
            'sim', 
            default_value='true', 
            description='Run in simulation mode'
        ),
        DeclareLaunchArgument(
            'camera_topic', 
            default_value='/image_raw', # Changed default to a more common topic
            description='Camera topic for the AI node'
        ),
    ]

    # Get the package share directories
    dexhand_ai_pkg = FindPackageShare('dexhand_ai')
    dexhand_description_pkg = FindPackageShare('dexhand_description')

    # Get paths to config files
    ai_config_file = PathJoinSubstitution([dexhand_ai_pkg, 'config', 'object_gesture_map.yaml'])
    rviz_config_file = PathJoinSubstitution([dexhand_description_pkg, 'rviz', 'dexhand_view.rviz'])

    # --- Nodes to launch ---

    # 1. Include the launch file from dexhand_description
    # This will start the RobotStatePublisher and load the URDF
    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([dexhand_description_pkg, 'launch', 'dexhand_description.launch.py'])
        ),
        launch_arguments={'use_sim_time': LaunchConfiguration('sim')}.items()
    )

    # 2. AI Gesture Bridge Node
    ai_gesture_bridge_node = Node(
        package='dexhand_ai',
        executable='ai_gesture_bridge',
        name='ai_bridge',
        parameters=[
            {'config_path': ai_config_file},
            {'camera_topic': LaunchConfiguration('camera_topic')}
        ],
        output='screen'
    )
    
    # 3. DexHand Gesture Controller Node
    gesture_controller_node = Node(
        package='dexhand_gesture_controller',
        executable='gesture_controller',
        name='gesture_controller',
        parameters=[{'simulation': LaunchConfiguration('sim')}],
        output='screen'
    )
    
    # 4. RViz2 Node (only launches if 'sim' is true)
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        condition=IfCondition(LaunchConfiguration('sim'))
    )

    return LaunchDescription(declared_arguments + [
        robot_description_launch,
        ai_gesture_bridge_node,
        gesture_controller_node,
        rviz_node
    ])
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
import cv2
import json
import time
import os
from datetime import datetime

class LearningCollector(Node):
    def __init__(self):
        super().__init__('learning_collector')
        self.bridge = CvBridge()
        
        # Data storage
        self.dataset = []
        self.current_session = None
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/color/image_raw',
            self.image_callback, 10
        )
        self.gesture_sub = self.create_subscription(
            String, '/dexhand_gesture',
            self.gesture_callback, 10
        )
        self.joints_sub = self.create_subscription(
            JointState, '/joint_states',
            self.joints_callback, 10
        )
        
        # State
        self.latest_image = None
        self.latest_gesture = None
        self.latest_joints = None
        
        # Create data directory
        os.makedirs('demonstrations', exist_ok=True)
        
        self.get_logger().info("Learning Collector ready. Collecting demonstrations...")

    def image_callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def gesture_callback(self, msg):
        self.latest_gesture = msg.data
        self.record_demo()

    def joints_callback(self, msg):
        self.latest_joints = list(msg.position)

    def record_demo(self):
        """Record a demonstration sample"""
        if all([self.latest_image, self.latest_gesture, self.latest_joints]):
            sample = {
                'timestamp': datetime.now().isoformat(),
                'gesture': self.latest_gesture,
                'joint_positions': self.latest_joints,
                'image_shape': self.latest_image.shape
            }
            
            # Save image separately
            img_path = f"demonstrations/img_{int(time.time())}.jpg"
            cv2.imwrite(img_path, self.latest_image)
            sample['image_path'] = img_path
            
            self.dataset.append(sample)
            
            # Save to JSON
            with open('demonstrations/dataset.json', 'w') as f:
                json.dump(self.dataset, f, indent=2)
            
            self.get_logger().info(
                f"Recorded: {self.latest_gesture} "
                f"(Total samples: {len(self.dataset)})"
            )

    def save_dataset(self):
        """Save complete dataset"""
        filename = f"demonstrations/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.dataset, f, indent=2)
        self.get_logger().info(f"Dataset saved to {filename}")

def main(args=None):
    rclpy.init(args=args)
    node = LearningCollector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_dataset()
        node.destroy_node()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import cv2
from train_policy import PolicyNet
import os

class PolicyDeploy(Node):
    def __init__(self, model_path='dexhand_policy.pth'):
        super().__init__('policy_deploy')
        self.bridge = CvBridge()
        
        # Load trained model
        self.model = PolicyNet()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Gesture map (reverse of training)
        self.id_to_gesture = {
            0: 'reset', 1: 'fist', 2: 'point',
            3: 'peace', 4: 'open_hand', 5: 'wave'
        }
        
        # Publishers
        self.gesture_pub = self.create_publisher(
            String, '/dexhand_gesture', 10
        )
        
        # Subscribers
        self.camera_sub = self.create_subscription(
            Image, '/camera/color/image_raw',
            self.camera_callback, 10
        )
        
        self.get_logger().info("Policy deployment ready!")

    def camera_callback(self, msg):
        try:
            # Convert image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Preprocess
            image = cv2.resize(cv_image, (224, 224))
            image = torch.from_numpy(image).float().permute(2,0,1).unsqueeze(0)
            image = image / 255.0
            
            # Predict
            with torch.no_grad():
                outputs = self.model(image)
                pred_id = torch.argmax(outputs, dim=1).item()
                gesture = self.id_to_gesture[pred_id]
            
            # Publish
            msg = String()
            msg.data = gesture
            self.gesture_pub.publish(msg)
            
            self.get_logger().info(f"AI Policy: {gesture}")
            
        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = PolicyDeploy()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
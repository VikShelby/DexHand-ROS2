#!/usr/bin/env python3
"""
dexhand_detect_viz_node.py

A ROS2 node that:
 - subscribes to an Image topic (default: /camera/color/image_raw)
 - falls back to the local webcam if no images arrive from ROS
 - runs YOLOv5 (torch.hub) for object detection
 - runs MediaPipe Pose, Hands, FaceMesh
 - draws boxes, labels and skeletons
 - publishes gestures (std_msgs/String) mapped from detected objects
 - shows an OpenCV window when DISPLAY is available (WSL-aware fallback)
 - keyboard controls: q=quit, o=toggle objects, p=toggle pose, h=toggle hands, f=toggle face, s=screenshot
"""

import os
import time
import threading
import yaml
import numpy as np
import cv2

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy # <-- NEW LINE
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ament_index_python.packages import get_package_share_directory

# try import torch/mediapipe and give friendly error if missing
try:
    import torch
except Exception as e:
    raise RuntimeError("Torch not found. Install with: pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118") from e

try:
    import mediapipe as mp
except Exception as e:
    raise RuntimeError("MediaPipe not found. Install with: pip install mediapipe") from e

# -----------------------
# Helper drawing funcs
# -----------------------
FONT = cv2.FONT_HERSHEY_SIMPLEX

def normalized_to_pixel_coords(norm_x, norm_y, image_width, image_height):
    x_px = min(max(int(norm_x * image_width + 0.5), 0), image_width - 1)
    y_px = min(max(int(norm_y * image_height + 0.5), 0), image_height - 1)
    return x_px, y_px

def draw_yolo_results(frame, preds, names):
    # preds.xyxy[0] is tensor (N,6) [x1,y1,x2,y2,conf,cls]
    if preds is None: return
    arr = preds.xyxy[0].cpu().numpy()
    for det in arr:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = names[int(cls)] if names and int(cls) in names else str(int(cls))
        text = f"{label} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (10, 255, 10), 2)
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), (10, 255, 10), -1)
        cv2.putText(frame, text, (x1, y1 - 4), FONT, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

# -----------------------
# Node
# -----------------------
class DexHandDetectViz(Node):
    def __init__(self):
        super().__init__('dexhand_detect_viz')

        # parameters
        pkg_share = ''
        try:
            pkg_share = get_package_share_directory('dexhand_ai')
        except Exception:
            pkg_share = os.path.join(os.getcwd(), 'config')  # fallback
        default_cfg = os.path.join(pkg_share, 'config', 'object_gesture_map.yaml')

        self.declare_parameter('config_path', default_cfg)
        self.declare_parameter('camera_topic', '/camera/color/image_raw')
        self.declare_parameter('gesture_topic', '/dexhand_gesture')
        self.declare_parameter('yolo_model', 'yolov5s')
        self.declare_parameter('confidence_threshold', 0.35)
        self.declare_parameter('use_local_webcam_if_no_ros', True)
        self.declare_parameter('webcam_index', 0)
        self.declare_parameter('frame_width', 1280)
        self.declare_parameter('frame_height', 720)

        cfg_path = self.get_parameter('config_path').get_parameter_value().string_value
        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        gesture_topic = self.get_parameter('gesture_topic').get_parameter_value().string_value
        yolo_model = self.get_parameter('yolo_model').get_parameter_value().string_value
        conf_thresh = float(self.get_parameter('confidence_threshold').get_parameter_value().double_value)
        self.use_local_webcam = bool(self.get_parameter('use_local_webcam_if_no_ros').get_parameter_value().bool_value)
        self.webcam_index = int(self.get_parameter('webcam_index').get_parameter_value().integer_value)
        self.frame_w = int(self.get_parameter('frame_width').get_parameter_value().integer_value)
        self.frame_h = int(self.get_parameter('frame_height').get_parameter_value().integer_value)

        # load mapping file
        try:
            with open(cfg_path, 'r') as f:
                cfg = yaml.safe_load(f)
                self.object_gesture_map = cfg.get('object_gesture_map', cfg)
        except Exception:
            self.get_logger().warn(f"Could not open config {cfg_path}, using minimal default mapping.")
            self.object_gesture_map = {'default': 'reset', 'person': 'wave'}

        # CvBridge
        self.bridge = CvBridge()

        # load YOLOv5
        self.get_logger().info(f"Loading YOLOv5 model: {yolo_model} ...")
        try:
            self.yolo = torch.hub.load('ultralytics/yolov5', yolo_model, pretrained=True, verbose=False)
            self.yolo.conf = conf_thresh
            self.yolo.iou = 0.45
            self.get_logger().info("YOLO loaded.")
        except Exception as e:
            self.get_logger().fatal(f"Failed to load YOLOv5: {e}")
            raise

        # MediaPipe setup
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_mesh
        self.pose_detector = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hands_detector = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.face_detector = self.mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                   min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # --- THE FIX IS HERE ---
        # Create a QoS profile that matches the camera's BEST_EFFORT policy
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        ) # <-- NEW BLOCK

        # publishers & subscriptions
        self.gesture_pub = self.create_publisher(String, gesture_topic, 10)
        # Use the new qos_profile for the subscription
        self.img_sub = self.create_subscription(Image, camera_topic, self.ros_image_cb, qos_profile) # <-- MODIFIED LINE

        # internal state
        self.last_sent_gesture = "reset"
        self.is_camera_live = False
        self.latest_frame = None
        self.window_name = "DexHand AI Vision"
        self.lock = threading.Lock()

        # toggles
        self.do_yolo = True
        self.do_pose = True
        self.do_hands = True
        self.do_face = True

        # start display thread
        self.display_ok = self._check_display_available()
        self.get_logger().info(f"Display available: {self.display_ok}")
        self.display_thread = threading.Thread(target=self.display_loop, daemon=True)
        self.display_thread.start()

        # timer heartbeat
        self.create_timer(2.0, self.heartbeat)

        # optionally open webcam fallback after short delay if no ROS images received
        if self.use_local_webcam:
            self.create_timer(2.5, self._start_webcam_if_needed)

        self.get_logger().info("DexHand detect/viz node ready.")

    # ---------- ROS callback ----------
    def ros_image_cb(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.lock:
                self.latest_frame = cv_image.copy()
            self.is_camera_live = True
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge error: {e}")

    # ---------- webcam fallback ----------
    def _start_webcam_if_needed(self):
        if self.is_camera_live or not self.use_local_webcam:
            return
        # spawn a webcam reader thread
        self.get_logger().info("No ROS camera frames detected — starting local webcam fallback.")
        self.webcam_cap = cv2.VideoCapture(self.webcam_index, cv2.CAP_V4L2)
        self.webcam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_w)
        self.webcam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_h)
        if not self.webcam_cap.isOpened():
            self.get_logger().warn("Webcam fallback not available or cannot be opened.")
            return
        def webcam_reader():
            while rclpy.ok():
                ret, frame = self.webcam_cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                with self.lock:
                    self.latest_frame = frame.copy()
                self.is_camera_live = True
                time.sleep(0.01)
        t = threading.Thread(target=webcam_reader, daemon=True)
        t.start()

    # ---------- processing & display loop ----------
    def display_loop(self):
        screenshot_counter = 0
        while rclpy.ok():
            frame = None
            with self.lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()

            if frame is None:
                time.sleep(0.02)
                continue

            # keep original for later publishing & overlays
            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # YOLO
            yolo_results = None
            if self.do_yolo:
                try:
                    # pass RGB so yolov5 doesn't re-color wrong
                    yolo_results = self.yolo(frame_rgb, size=640)
                    draw_yolo_results(frame, yolo_results, self.yolo.names)
                except Exception as e:
                    self.get_logger().error(f"YOLO inference error: {e}")

            # MediaPipe Pose
            if self.do_pose:
                try:
                    pose_results = self.pose_detector.process(frame_rgb)
                    if pose_results.pose_landmarks:
                        self.mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                                       landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255,255,0), thickness=2, circle_radius=2),
                                                       connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,255,255), thickness=2))
                        # annotate nose
                        nose = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
                        nx, ny = normalized_to_pixel_coords(nose.x, nose.y, w, h)
                        cv2.putText(frame, "Nose", (nx+5, ny+5), FONT, 0.5, (255,255,0), 1, cv2.LINE_AA)
                except Exception as e:
                    self.get_logger().error(f"Pose error: {e}")

            # MediaPipe Hands
            if self.do_hands:
                try:
                    hands_results = self.hands_detector.process(frame_rgb)
                    if hands_results.multi_hand_landmarks:
                        for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
                            self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                                                           landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255,0,255), thickness=2, circle_radius=2),
                                                           connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(200,0,200), thickness=2))
                            xs = [lm.x for lm in hand_landmarks.landmark]
                            ys = [lm.y for lm in hand_landmarks.landmark]
                            cx, cy = normalized_to_pixel_coords(np.mean(xs), np.mean(ys), w, h)
                            label = handedness.classification[0].label
                            cv2.putText(frame, label, (cx-20, cy-10), FONT, 0.7, (255,0,255), 2, cv2.LINE_AA)
                except Exception as e:
                    self.get_logger().error(f"Hands error: {e}")

            # MediaPipe Face
            if self.do_face:
                try:
                    face_results = self.face_detector.process(frame_rgb)
                    if face_results.multi_face_landmarks:
                        for face_landmarks in face_results.multi_face_landmarks:
                            self.mp_drawing.draw_landmarks(frame, face_landmarks, self.mp_face.FACEMESH_TESSELATION,
                                                           landmark_drawing_spec=None,
                                                           connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))
                except Exception as e:
                    self.get_logger().error(f"Face error: {e}")

            # Gesture mapping from top object detected (if YOLO active)
            current_gesture = self.object_gesture_map.get('default', 'reset')
            detected_label = 'none'
            if self.do_yolo and yolo_results is not None and len(yolo_results.xyxy[0]) > 0:
                top = yolo_results.xyxy[0][0]  # first detection
                cls_idx = int(top[5].cpu().numpy()) if hasattr(top[5], 'cpu') else int(top[5])
                detected_label = self.yolo.names[cls_idx] if cls_idx in self.yolo.names else str(cls_idx)
                current_gesture = self.object_gesture_map.get(detected_label, self.object_gesture_map.get('default', 'reset'))

            # publish gesture if changed
            if current_gesture != self.last_sent_gesture:
                msg = String()
                msg.data = current_gesture
                self.gesture_pub.publish(msg)
                self.get_logger().info(f"OBJECT: {detected_label} -> ACTION: {current_gesture}")
                self.last_sent_gesture = current_gesture

            # HUD overlay
            fps_text = f"CAMERA LIVE" if self.is_camera_live else "NO CAMERA"
            cv2.putText(frame, fps_text, (10, 30), FONT, 0.8, (0,255,0) if self.is_camera_live else (0,0,255), 2)
            cv2.putText(frame, f"GESTURE: {self.last_sent_gesture.upper()}", (10, 60), FONT, 0.8, (0,255,255), 2)

            # attempt to show window, else fallback to saving frames periodically
            if self.display_ok:
                try:
                    cv2.imshow(self.window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.get_logger().info("Quit key pressed.")
                        rclpy.shutdown()
                        break
                    elif key == ord('o'):
                        self.do_yolo = not self.do_yolo
                        self.get_logger().info(f"Toggle YOLO -> {self.do_yolo}")
                    elif key == ord('p'):
                        self.do_pose = not self.do_pose
                    elif key == ord('h'):
                        self.do_hands = not self.do_hands
                    elif key == ord('f'):
                        self.do_face = not self.do_face
                    elif key == ord('s'):
                        fname = f"screenshot_{screenshot_counter}.png"
                        cv2.imwrite(fname, frame)
                        self.get_logger().info(f"Saved {fname}")
                        screenshot_counter += 1
                except Exception as e:
                    self.get_logger().warn(f"Display failed during show: {e}")
                    self.display_ok = False
            else:
                # minimal fallback: save a frame every N sec so you can inspect later
                if screenshot_counter % 150 == 0:
                    fname = f"frame_fallback_{screenshot_counter}.png"
                    cv2.imwrite(fname, frame)
                    self.get_logger().info(f"No display available: saved fallback frame {fname}")
                screenshot_counter += 1
                time.sleep(0.02)

        # cleanup
        try:
            self.pose_detector.close()
            self.hands_detector.close()
            self.face_detector.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def heartbeat(self):
        if not self.is_camera_live:
            self.get_logger().warn("Heartbeat: waiting for camera frames...")
        else:
            self.get_logger().info("Heartbeat: camera live.")

    def _check_display_available(self):
        """
        Heuristic: check if DISPLAY or /mnt/wslg exists and X/Wayland can be opened.
        This does not attempt to open GL; it simply tries a tiny namedWindow test.
        """
        # quick environment clues
        disp = os.environ.get('DISPLAY', '')
        way = os.environ.get('WAYLAND_DISPLAY', '')
        if '/mnt/wslg' in os.listdir('/') or os.path.exists('/mnt/wslg'):
            # try to create a tiny window to verify
            try:
                cv2.namedWindow("__wsl_test__", cv2.WINDOW_NORMAL)
                cv2.imshow("__wsl_test__", np.zeros((10,10,3), dtype=np.uint8))
                cv2.waitKey(1)
                cv2.destroyWindow("__wsl_test__")
                return True
            except Exception as e:
                self.get_logger().warn(f"Display check failed: {e}")
                return False
        # fallback: if DISPLAY is set, try showing
        if disp:
            try:
                cv2.namedWindow("__disp_test__", cv2.WINDOW_NORMAL)
                cv2.imshow("__disp_test__", np.zeros((10,10,3), dtype=np.uint8))
                cv2.waitKey(1)
                cv2.destroyWindow("__disp_test__")
                return True
            except Exception as e:
                self.get_logger().warn(f"DISPLAY set but opencv can't open window: {e}")
                return False
        return False

def main(args=None):
    rclpy.init(args=args)
    node = DexHandDetectViz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down (KeyboardInterrupt)')
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()

if __name__ == '__main__':
    main()
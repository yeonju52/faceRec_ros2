# from typing import List, Dict

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

import message_filters
from cv_bridge import CvBridge

# from ultralytics import YOLO
# from ultralytics.engine.results import Results
# from ultralytics.engine.results import Boxes
# from ultralytics.engine.results import Masks
# from ultralytics.engine.results import Keypoints

from sensor_msgs.msg import Image
from yolov8_msgs.msg import Detection
from yolov8_msgs.msg import DetectionArray
from yolov8_msgs.msg import FaceID
from yolov8_msgs.msg import FaceIDArray


import sys
# sys.path.insert(0, "/home/lee52/ros2_ws/py310/lib/python3.10/site-packages") # 우선순위 지정
sys.path.append("/home/lee52/ros2_ws/src/yolov8_ros/faceRec")

from adaface import AdaFace

class AdafaceNode(Node):
    def __init__(self) -> None:
        super().__init__("adaface_node")
        self.get_logger().info('Adaface is now starting...')

        # params
        self.declare_parameter("fr_weight", "ir_50")
        model = self.get_parameter("fr_weight").get_parameter_value().string_value

        self.declare_parameter("device", "cuda:0")
        self.device = self.get_parameter("device").get_parameter_value().string_value

        self.declare_parameter("option", 1)  
        option = self.get_parameter("option").get_parameter_value().integer_value
        
        self.declare_parameter("thresh", 0.2)
        self.thresh = self.get_parameter("thresh").get_parameter_value().double_value

        self.declare_parameter("max_obj", 6)        
        self.max_obj = self.get_parameter("max_obj").get_parameter_value().integer_value

        self.declare_parameter("dataset", "face_dataset/test")  
        self.dataset = self.get_parameter("dataset").get_parameter_value().string_value
        
        self.declare_parameter("video", "0")
        self.video = self.get_parameter("video").get_parameter_value().string_value
        
        self.declare_parameter("image_reliability",
                               QoSReliabilityPolicy.BEST_EFFORT)
        image_qos_profile = QoSProfile(
            reliability=self.get_parameter(
                "image_reliability").get_parameter_value().integer_value,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self.cv_bridge = CvBridge()
        self.adaface = AdaFace(
            model=model,
            option=option,
            dataset=self.dataset,
            video=self.video,
            max_obj=self.max_obj,
            thresh=self.thresh,
        )
        self.get_logger().info('Adaface load suceess...')
        # if adaface.option == 0:
        #     adaface.store_embedding()
        # elif adaface.option == 1:
        #     adaface.run_video()
        # else:
        #     print("Error: 잘못된 argument 입력")
        #     sys.exit(1)

        # pubs
        self._pub = self.create_publisher(FaceIDArray, "recogntion", 10)

        # subs
        image_sub = message_filters.Subscriber(
            self, Image, "image_raw", qos_profile=image_qos_profile)
        # self._sub = self.create_subscription(
        #     Image, "image_raw", self.image_cb,
        #     image_qos_profile
        # )
        tracking_sub = message_filters.Subscriber(
            self, DetectionArray, "detections", qos_profile=10)
        
        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (image_sub, tracking_sub), 10, 0.5)
        self._synchronizer.registerCallback(self.recogntion_cb)

    def recogntion_cb(self, img_msg: Image, detections_msg: DetectionArray) -> None:
        recognized_faces_msg = FaceIDArray()
        recognized_faces_msg.header = img_msg.header

        # convert image
        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg)
        # 로그 추가
        self.get_logger().info('Received image message: %s', img_msg)
        self.get_logger().info('Received detections message: %s', detections_msg)
        self.get_logger().info('Converted image to cv_image: %s', cv_image)
        # recogntion_list = []
        # recogntion: FaceID
        # for detection in detections_msg.detections:
        #     recogntion_list.append(
        #         [
        #             detection.bbox.center.position.x - detection.bbox.size.x / 2,
        #             detection.bbox.center.position.y - detection.bbox.size.y / 2,
        #             detection.bbox.center.position.x + detection.bbox.size.x / 2,
        #             detection.bbox.center.position.y + detection.bbox.size.y / 2,
        #             detection.score,
        #             detection.class_id
        #         ]
        #     )

def main():
    rclpy.init()
    node = AdafaceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

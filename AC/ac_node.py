import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from std_msgs.msg import String
from mad_dog_interface.srv import ActivatePatrol
from mad_dog_interface.srv import ActivateGoHome
from mad_dog_interface.action import NavigateToSuspect
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point #PoseWithCovarianceStamped
from cv_bridge import CvBridge
from ultralytics import YOLO  # YOLOv8
import numpy as np
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
import cv2
import json
import torch
from geometry_msgs.msg import PoseStamped

class MoveToZoneActionServer(Node):

    def __init__(self):
        super().__init__('move_to_zone_action_server')

        self.cap = cv2.VideoCapture(4)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        
        # YOLOv8 모델 초기화
        self.yolo_model = YOLO("./yolov8n.pt")

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,  # 데이터 손실 없이 안정적으로 전송
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, # 구독자가 서버와 연결된 후 그 동안 수집된 데이터를 받을 수 있음
            history=QoSHistoryPolicy.KEEP_LAST, # 최근 메시지만 유지
            depth=10  # 최근 10개의 메시지를 유지
        )  
        
        # DA zone string
        self.zone_action_server = ActionServer(
            NavigateToSuspect,  
            'navigate_to_zone', 
            self.zone_callback 
        )

        # SM patrol 변경값 request
        self.patrol_service = self.create_service(ActivatePatrol,
            'patrol_service',
            self.handle_patrol_service_request
        )

        # SM gohome request
        self.gohome_service = self.create_service(ActivateGoHome,
            'gohome_service',
            self.handle_gohome_service_request
        )
    
                
        # SM AMR_Image pub
        self.AMR_image_publisher = self.create_publisher(
            Image,
            'amr_image',
            self.image_callback,
            qos_profile
        )
        
        # AMR AMR_Image_sub
        self.AMRcam_image_subscribtion = self.create_subscription(
            Image,
            'amrcam_image', 
            self.amr_image_callback,
            10
        )

        # AMR goal좌표 pub
        self.publisher_ = self.create_publisher(
            PoseStamped,
            '/move_base_simple/goal',
            10
        )

        
        
        self.img_timer = self.create_timer(0.1, self.image_callback)

        # 정지상태
        self.AMR_mode = 0

        # 5개 영역의 하나의 꼭짓점 좌표 정의 (각 영역별로 임의의 path를 지정해주는 로직)
        self.zones = {
            'A': Point(x=1.0, y=2.0),
            'B': Point(x=3.0, y=4.0),
            'C': Point(x=5.0, y=6.0),
            'D': Point(x=7.0, y=8.0),
            'E': Point(x=9.0, y=10.0)
        }

    
    # zond에 대한 str을 받음
    def zone_callback(self, goal_handle):
        # goal 받았다는 로그
        self.get_logger().info(f"Received goal: {goal_handle.request.area}")

        if self.AMR_mode == 1:
            # mode가 1인 경우: Goal 값으로 이동
            for zone in goal_handle.request.area:
                self.get_logger().info(f"Navigating to zone: {zone}")
                # AMR 에 send goal 할 수 있도록 action 설계 추가

            result = NavigateToSuspect.Result()
            result.success = True
            goal_handle.succeed()

    # 디지털 맵 내 지정 구역으로 이동
    def navigate_to_zone(self, zone_name):
        
        # 각 zone에 대한 좌표를 지정 (예시: zone_name에 대응하는 좌표)
        zone_coordinates = {
            'zone_1': {'x': 1.0, 'y': 2.0},
            'zone_2': {'x': 3.0, 'y': 4.0},
            'zone_3': {'x': 5.0, 'y': 6.0},
        }

        if zone_name in self.zones:
            # 목표 좌표 가져오기
            target_pose = self.zones[zone_name]

            # 목표 좌표를 PoseStamped 메시지로 생성
            goal_msg = PoseStamped()
            goal_msg.header.frame_id = 'map'  # SLAM에서 사용되는 좌표계 (보통 'map' 프레임)
            goal_msg.pose.position.x = target_pose['x']
            goal_msg.pose.position.y = target_pose['y']
            goal_msg.pose.orientation.w = 1.0  # 회전 값 (회전 없음)

            # 목표를 move_base_simple/goal로 발행
            self.goal_publisher.publish(goal_msg)
            self.get_logger().info(f"Sending goal to zone {zone_name}: {target_pose}")
    
    def handle_patrol_service_request(self, request, response):
        response.success = True
        if self.AMR_mode == 0:
            self.AMR_mode = 1
        return response

    def handle_gohome_service_request(self, request, response):
        response.success = True
        self.AMR_mode = 0
        # 집으로 가는 함수
        return response
    
    def amr_image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.amr_image_frame = frame

    def image_callback(self):
        
        # 객체 감지 및 추적
        try:
            detections, track_ids, class_names, boxes, confidences = self.detect_objects()
        except:
            return False
        
        ros_image = self.convert_cv2_to_ros_image(detections)
        # ros_image.header = msg.Header()
        # ros_image.header.stamp = self.get_clock().now().to_msg()
        self.image_publisher.publish(ros_image)
        self.get_logger().info("Publishing video frame")

        detect_dic = {}

        for track_id, class_name, box, confidence in zip(track_ids, class_names, boxes, confidences):
            # Tensor를 리스트로 변환
            if isinstance(box, torch.Tensor):  # box가 Tensor인 경우
                box_list = box.tolist()  # Tensor를 리스트로 변환
            else:
                box_list = box  # 이미 리스트라면 그대로 사용

            if isinstance(confidence, torch.Tensor):  # confidence가 Tensor인 경우
                confidence_value = confidence.item()  # Tensor를 float으로 변환
            else:
                confidence_value = confidence  # 이미 숫자라면 그대로 사용
            
            # detect_dic에 정보 저장
            detect_dic[track_id] = {'class': class_name,'box': box_list, 'confidence': confidence_value}
            
            if confidence_value < 0.5:
                zone_name = self.get_zone_for_detection(track_id, box)
                if zone_name:
                    # 현재 구역과 이전 구역이 다르면 목표 변경
                    if self.detected_objects.get(track_id) != zone_name:
                        self.send_amr_request(zone_name)
                        self.detected_objects[track_id] = zone_name

        # JSON 직렬화
        detect_str = json.dumps(detect_dic)

        # ROS2 메시지로 포장
        detect_msg = String()
        detect_msg.data = detect_str

        # 퍼블리시
        self.detection_publisher.publish(detect_msg)


    # ROS2 이미지를 OpenCV로 변환
    def convert_cv2_to_ros_image(self, ros_image):
        # ROS2 이미지를 OpenCV로 변환
        bridge = CvBridge()
        return bridge.cv2_to_imgmsg(ros_image, encoding='bgr8')
    
    # 객체 탐색 기능
    # 객체를 탐색하여 해당 객체의 바운딩 박스 좌표와 confidencefmf detection에 저장
    def detect_objects(self):
        # Store the track history
        
        success, frame = self.cap.read()
        # YOLOv8 객체 탐지
        results = self.yolo_model.track(frame, persist=True)
        # Get the boxes and track IDs
        class_ids = results[0].boxes.cls
        class_names = [self.yolo_model.names[int(class_id)] for class_id in class_ids]
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = self.track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)
                # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
        

        return annotated_frame, track_ids, class_names, boxes, confidences
   

        

def main(args=None):
    rclpy.init(args=args)
    action_server = MoveToZoneActionServer()
    rclpy.spin(action_server)  # 노드가 계속 실행되도록

    rclpy.shutdown()

if __name__ == '__main__':
    main()

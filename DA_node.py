import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import torch
import cv2
from ultralytics import YOLO  # YOLOv8
from deep_sort_realtime.deepsort_tracker import DeepSort
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from action_interfaces.action import NavigateToCoordinate

class AMRControlNode(Node):
    def __init__(self):
        super().__init__('amr_control_node')

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 10)

        # YOLOv8 모델 초기화
        self.yolo_model = YOLO("yolov8.pt") 
        self.deepsort = DeepSort()
        
        # 실시간 처리 및 손실 없는 전송을 위해 설정
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,  # 데이터 손실 없이 안정적으로 전송
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, # 구독자가 서버와 연결된 후 그 동안 수집된 데이터를 받을 수 있음
            history=QoSHistoryPolicy.KEEP_LAST, # 최근 메시지만 유지
            depth=10  # 최근 10개의 메시지를 유지
        )       

        # 카메라 스트리밍 토픽생성
        self.image_publisher = self.create_subscription(
            Image,
            '/webcam/image_raw',
            self.image_callback,
            qos_profile
        )

        # AMR 목표로 보낼 action client생성
        self._action_client = ActionClient(self, NavigateToCoordinate, 'navigate_to_coordinate')

        # 5개 영역의 하나의 꼭짓점 좌표 정의 (각 영역별로 임의의 path를 지정해주는 로직)
        self.zones = {
            'A': Point(x=1.0, y=2.0),
            'B': Point(x=3.0, y=4.0),
            'C': Point(x=5.0, y=6.0),
            'D': Point(x=7.0, y=8.0),
            'E': Point(x=9.0, y=10.0)
        }

    
    # 객체를 탐지하여 confidence를 통해 
    # 객체를 탐지하여 탐지된 객체의 confidence가 0.5이하인 경우 출입 영역의 이름을 amr에 요청 
    def image_callback(self):

        ret, msg = self.cap.read()
        # 웹캠 이미지 받기
        frame = self.convert_ros_image_to_cv2(msg)
        
        # 객체 감지 및 추적
        detections = self.detect_objects(frame)
        
        for detection in detections:
            confidence = detection['confidence']
            if confidence < 0.5:
                zone_name = self.get_zone_for_detection(detection)
                self.send_amr_request(zone_name)

    # ROS2 이미지를 OpenCV로 변환
    def convert_ros_image_to_cv2(self, ros_image):
        # ROS2 이미지를 OpenCV로 변환
        bridge = CvBridge()
        return bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')
    
    # 객체 탐색 기능
    # 객체를 탐색하여 해당 객체의 바운딩 박스 좌표와 confidencefmf detection에 저장
    def detect_objects(self, frame):
        # YOLOv8 객체 탐지
        results = self.yolo_model(frame)
        detections = []
        for result in results:
            bbox = result['bbox']
            confidence = result['confidence']
            detections.append({'bbox': bbox, 'confidence': confidence})

        # Deep SORT 추적
        track_results = self.deepsort.update_tracks(detections, frame)  # 트랙 결과
        return detections

    # 객체의 바운딩 박스의 중앙점이 지정된 zone안에 있다면 해당 지역의 이름을 return
    def get_zone_for_detection(self, detection):
        # 객체의 위치에 맞는 영역 결정 (예시로, 중간점을 기준으로 비교)
        x_center = (detection['bbox'][0] + detection['bbox'][2]) / 2
        y_center = (detection['bbox'][1] + detection['bbox'][3]) / 2
        
        # zone_name 별로 임의의 zone_coords를 지정해둔다.
        for zone_name, zone_coords in self.zones.items():
            if self.is_within_zone(x_center, y_center, zone_coords):
                return zone_name
        return None

    # 영역 내 포함 여부 확인 (SLAM 디지털 맵을 통해 범위 값을 최적화할 예정)
    def is_within_zone(self, x, y, zone_coords):
        # 영역의 좌표를 x1,x2, y1, y2라고 할때 x1-0.5, x2+0.5 범위 안에 객체의 중앙x 좌표가 있는지, y1-0.5,y2+0.5 범위 안에 객체의 중앙 y 좌표가 있는지 확인
        return zone_coords.x - 0.5 < x < zone_coords.x + 0.5 and zone_coords.y - 0.5 < y < zone_coords.y + 0.5

    # send amr에 action으로 이동 요청(action은 요청을 수행하는 도중 goal을 바꿀 수 있기 때문)
    def send_amr_request(self, zone_name):
        # zone_name이 주어 진다면
        if zone_name:
            self.get_logger().info(f"Sending AMR to zone {zone_name}")
            goal = NavigateToCoordinate.Goal()
            goal.x = self.zones[zone_name].x # zone_name과 매칭 되어 있는 디지털 맵 위의 x좌표
            goal.y = self.zones[zone_name].y # zone_name과 매칭 되어 있는 디지털 맵 위의 x좌표
            goal.zone_name = zone_name

            self._action_client.wait_for_server()

            # 요청 보내기
            self._action_client.send_goal_async(goal, feedback_callback=self.feedback_callback)

    def feedback_callback(self, feedback):
        # AMR의 피드백 처리 (진행 상태)
        self.get_logger().info(f"AMR Feedback: {feedback.status}")

def main(args=None):
    rclpy.init(args=args)
    node = AMRControlNode()

    rclpy.spin(node)

    rclpy.shutdown()

if __name__ == '__main__':
    main()

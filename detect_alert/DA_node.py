import rclpy
from rclpy.node import Node
from collections import defaultdict
from rclpy.action import ActionClient
from sensor_msgs.msg import Image
import std_msgs.msg as msg
from geometry_msgs.msg import Point, PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO  # YOLOv8
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from nav2_msgs.action import NavigateToPose  # Change here
import json
import torch


class AMRControlNode(Node):
    def __init__(self):
        super().__init__('amr_control_node')

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # YOLOv8 모델 초기화
        self.yolo_model = YOLO("./yolo_models/cctv.pt") 
        
        # 실시간 처리 및 손실 없는 전송을 위해 설정
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,  # 데이터 손실 없이 안정적으로 전송
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, # 구독자가 서버와 연결된 후 그 동안 수집된 데이터를 받을 수 있음
            history=QoSHistoryPolicy.KEEP_LAST, # 최근 메시지만 유지
            depth=10  # 최근 10개의 메시지를 유지
        )

        # 카메라 스트리밍 토픽생성
        self.image_publisher = self.create_publisher(
            Image,
            'cctv_image',
            qos_profile
        )
        
        self.detection_publisher = self.create_publisher(
            msg.String,
            'da_track_data',
            qos_profile
        )
        
        self.da_alert_publisher = self.create_publisher(
            msg.Bool,
            'da_alert',
            qos_profile
        )

        self.img_timer = self.create_timer(0.1, self.image_callback)
        self.alert_timer = self.create_timer(0.5, self.da_alert_callback)

        # AMR 목표로 보낼 prblisher
        self.area_publisher = self.create_publisher(
            msg.String,
            'navigate_to_zone',
            qos_profile
        )  # Change here

        # 5개 영역의 하나의 꼭짓점 좌표 정의 (각 영역별로 임의의 path를 지정해주는 로직)
    
        zones = {
            'A': [(3, 3), (4, 237), (195, 239), (196, 2)],
            'B': [(195, 238), (387, 236), (387, 1), (197, 4)],
            'C': [(388, 0), (604, 0), (585, 228), (388, 237)],
            'D': [(79, 240), (84, 478), (267, 479), (263, 238)],
            'E': [(264, 240), (269, 478), (453, 478), (459, 237)],
            'F': [(458, 236), (453, 477), (638, 479), (638, 231)]
        }
        self.zones = self.get_polygon_points(zones)

        self.detected_objects = defaultdict(lambda: None)
        self.track_history = defaultdict(lambda: [])
        self.detect_area = 'Not Found'
        self.isDetected = False
    
    def da_alert_callback(self):
        alertMsg = msg.Bool()
        alertMsg.data = self.isDetected
        self.da_alert_publisher.publish(alertMsg)
    
    def image_callback(self):
        
        # 객체 감지 및 추적
        try:
            detections, track_ids, class_names, boxes, confidences = self.detect_objects()
        except:
            return False
        
        # zone 그리기
        detections = self.draw_zones(detections)

        # 이미지 전송
        ros_image = self.convert_cv2_to_ros_image(detections)
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

        
        self.detected_objects = detect_dic
        
        # 거수자 분석 및 amr 명령
        self.detect_suspect()
        # JSON 직렬화
        detect_str = json.dumps(detect_dic)

        # ROS2 메시지로 포장
        detect_msg = msg.String()
        detect_msg.data = detect_str

        # 퍼블리시
        self.detection_publisher.publish(detect_msg)


    # ROS2 이미지를 OpenCV로 변환
    def convert_cv2_to_ros_image(self, ros_image):
        # ROS2 이미지를 OpenCV로 변환
        bridge = CvBridge()
        return bridge.cv2_to_imgmsg(ros_image, encoding='bgr8')
    
    def draw_zones(self, frame):
        for zone_name, points in self.zones.items():
            # NumPy 배열로 변환된 좌표에서 중심점 계산
            centroid_x = int(np.mean(points[:, 0, 0]))
            centroid_y = int(np.mean(points[:, 0, 1]))

            # 다각형 그리기
            cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)

            # 다각형 이름 표시
            cv2.putText(frame, zone_name, (centroid_x, centroid_y), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.6, color=(255, 0, 0), thickness=1)

        
        return frame
    
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

    def detect_suspect(self):
        for id, detected in self.detected_objects.items():
            if detected['class'] == 'dummy2':
                pos = self.get_middle_point_of_box(detected['box'])
                
                zone_name = self.get_zone_of_point(pos)
                if zone_name != None:
                    self.isDetected = True
                    self.send_amr_request(zone_name)
                    return
        
        self.isDetected = False

    def get_polygon_points(self, zones):
        return { k:np.array(points).reshape((-1, 1, 2)) for k,points in zones.items() }

    def get_middle_point_of_box(self,box):
        x_center = (box[0] + box[2] / 2)
        y_center = (box[1] + box[3] / 2)
        return (x_center, y_center)

    # 객체의 바운딩 박스의 중앙점이 지정된 zone안에 있다면 해당 지역의 이름을 return
    # def get_zone_for_detection(self, track_id, box):
    #     # 객체의 위치에 맞는 영역 결정 (예시로, 중간점을 기준으로 비교)
        
    #     object_dic = {}
    #     x_center = (box[0] + box[2] / 2)
    #     y_center = (box[1] + box[3] / 2)
    #     object_dic[track_id] = [x_center, y_center]
        
    #     # zone_name 별로 임의의 zone_coords를 지정해둔다.
    #     for zone_name, zone_coords in self.zones.items():
    #         if self.is_within_zone(object_dic[track_id][0], object_dic[track_id][1], zone_name):
    #             return zone_name
    #     return 'Not Found'
    def get_zone_of_point(self, point):
        for k, polygon_points in self.zones.items():
            result = cv2.pointPolygonTest(polygon_points, point, False)
            if result >= 0:
                return k
        return None

    # 영역 내 포함 여부 확인 (SLAM 디지털 맵을 통해 범위 값을 최적화할 예정)
    def is_within_zone(self, x, y, zone_name):
        # 영역의 좌표를 x1,x2, y1, y2라고 할때 x1-0.5, x2+0.5 범위 안에 객체의 중앙x 좌표가 있는지, y1-0.5,y2+0.5 범위 안에 객체의 중앙 y 좌표가 있는지 확인
        return self.zones[zone_name]['x1'] <= x <= self.zones[zone_name]['x2'] and self.zones[zone_name]['y1'] <= y <= self.zones[zone_name]['y2']

    # send amr에 action으로 이동 요청(action은 요청을 수행하는 도중 goal을 바꿀 수 있기 때문)
    def send_amr_request(self, zone_name):
        # zone_name이 주어 진다면
        if zone_name:
            self.get_logger().info(f"Sending AMR to zone {zone_name}")
            goal = msg.String()
            goal.data = zone_name
            self.area_publisher.publish(goal)

def main(args=None):
    rclpy.init(args=args)
    node = AMRControlNode()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

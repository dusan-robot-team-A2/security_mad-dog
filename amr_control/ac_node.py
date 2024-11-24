import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
import std_msgs.msg as msg
from mad_dog_interface.srv import ActivatePatrol
from mad_dog_interface.srv import ActivateGoHome
from mad_dog_interface.action import NavigateToSuspect
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Point, PoseStamped, Twist, Quaternion, PoseWithCovarianceStamped
from cv_bridge import CvBridge
from ultralytics import YOLO  # YOLOv8
import numpy as np
import cv2
from nav2_msgs.action import NavigateToPose, FollowWaypoints
import math
import threading
import sys
import select
import termios
import tty
###########################################
'''
Part: AC_tracking

수정사항

1. 많은 객체 중에 침입자를 결정 로직 재확인 필요 ( 고려중인 로직: 중앙값을 바라보고 터틀봇 프레임중앙과 가장 가까운 박스 중앙값을 갖는 객체를 침입자로)
2. gohome, waypoint 좌표값 정의 (need update)
3. cctv camera 상의 영역 폴리곤 나누고 중앙값 얻기 (need update)
4. area에 대응하는 디지털 맵 내의 좌표값 얻기(AMR 카메라를 통해 해당 영역을 가장 잘 관찰할 수 있으면서 해당 영역의 중앙값을 바라보도록 설계)


'''
###########################################

class MoveToZoneActionServer(Node):

    def __init__(self):
        super().__init__('move_to_zone_action_server')

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        # self.success, self.frame = self.cap.read()
        # # YOLOv8 객체 탐지
        # self.self.results = self.yolo_model.track(self.frame, persist=True)

        # YOLOv8 모델 초기화
        self.yolo_model = YOLO("./yolov8n.pt")

        # DA zone string
        self.zone_subscrib = self.create_subscription(msg.String,'navigate_to_zone', self.zone_callback, 10)


        # SM patrol 변경값 request
        self.patrol_service = self.create_service(ActivatePatrol,'patrol_service',self.handle_patrol_service_request)
        # SM gohome request
        self.gohome_service = self.create_service(ActivateGoHome,'gohome_service',self.handle_gohome_service_request)
        # SM AMR_Image pub
        self.AMR_image_publisher = self.create_publisher(Image,'amr_image',10)
        # SM_tracked_image_pub
        self.sm_tracked_image_publisher = self.create_publisher(Image, 'tracked_image', 10)


        # AMR AMR_Image_sub
        # self.AMRcam_image_subscribtion = self.create_subscription(Image,'amrcam_image', self.amr_image_callback,10)
        # AMR_initial_pose_pub
        self.amr_initialpose_publisher = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.publish_initial_pose()
        # AMR_waypoint_action_client
        self.amr_waypoint_action_client = ActionClient(self, FollowWaypoints, '/follow_waypoints')
        # AMR goal좌표 pub
        self.amr_goal_publisher_ = self.create_publisher(PoseStamped,'/move_base_simple/goal',10)
        # AMR_navgoal_action_client
        self.amr_navgoal_client = ActionClient(self, NavigateToPose, 'navigate_to_pose') 
        # AMR_cmd_pub
        self.amr_cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)


        self.zone = None
        self.img_timer = self.create_timer(0.1, self.image_callback)
        
        self.screen_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.screen_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.screen_area = self.screen_width * self.screen_height
        self.target_area_threshold = 0.8 * self.screen_area

        # yolo result
        self.results = None

        # 정지상태
        self.AMR_mode = 0
        self.isHome = True

        # 5개 영역의 하나의 꼭짓점 좌표 정의 (각 영역별로 임의의 path를 지정해주는 로직)
        self.zones = {
            'A': Point(x=-1.05, y=-0.25),
            'B': Point(x=-0.8, y=-0.3),
            'C': Point(x=-0.515, y=-0.24),
            'D': Point(x=-0.855, y=-0.675),
            'E': Point(x=-0.495, y=-0.625),
            'F': Point(x=-0.665, y=-0.65),
            'Home': Point(x=0.043317, y=0.033049)
        }

    def euler_to_quaternion(self, roll, pitch, yaw):
        # Convert Euler angles to a quaternion
        qx = math.sin(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) - math.cos(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
        qy = math.cos(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2)
        qz = math.cos(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2) - math.sin(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2)
        qw = math.cos(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
        return Quaternion(x=qx, y=qy, z=qz, w=qw)

    def waypoint_goal(self):
        # 세 개의 웨이포인트 정의
        waypoints = []

        # 첫 번째 웨이포인트
        waypoint1 = PoseStamped()
        waypoint1.header.stamp.sec = 0
        waypoint1.header.stamp.nanosec = 0
        waypoint1.header.frame_id = "map"  # 프레임 ID를 설정 (예: "map")
        waypoint1.pose.position.x = 0.36131
        waypoint1.pose.position.y = -0.1551
        waypoint1.pose.position.z = 0.0

        waypoint1_yaw = 0.0  # Target orientation in radians
        waypoint1.pose.orientation = self.euler_to_quaternion(0, 0, waypoint1_yaw)
        
        # waypoint1.pose.orientation.x = 0.0
        # waypoint1.pose.orientation.y = 0.0
        # waypoint1.pose.orientation.z = -0.9999865408184966
        # waypoint1.pose.orientation.w = 0.005188273494832019
        waypoints.append(waypoint1)

        # 두 번째 웨이포인트
        waypoint2 = PoseStamped()
        waypoint2.header.stamp.sec = 0
        waypoint2.header.stamp.nanosec = 0
        waypoint2.header.frame_id = "map"  # 프레임 ID를 설정 (예: "map")
        waypoint2.pose.position.x = 0.2981
        waypoint2.pose.position.y = -0.61095
        waypoint2.pose.position.z = 0.0
        
        waypoint2_yaw = 0.0  # Target orientation in radians
        waypoint2.pose.orientation = self.euler_to_quaternion(0, 0, waypoint2_yaw)
        
        # waypoint2.pose.orientation.x = 0.0
        # waypoint2.pose.orientation.y = 0.0
        # waypoint2.pose.orientation.z = -0.9999330665398213
        # waypoint2.pose.orientation.w = 0.01156989370173046
        waypoints.append(waypoint2)

        # FollowWaypoints 액션 목표 생성 및 전송
        goal_msg = FollowWaypoints.Goal()
        goal_msg.poses = waypoints

        # 서버 연결 대기
        self.amr_waypoint_action_client.wait_for_server()

        print('ss')
        # 목표 전송 및 피드백 콜백 설정
        self._send_goal_future = self.amr_waypoint_action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.waypoint_feedback_callback
        )
        self._send_goal_future.add_done_callback(self.waypoint_response_callback)

    def waypoint_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Current Waypoint Index: {feedback.current_waypoint}')
    
    def waypoint_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')
        self._goal_handle = goal_handle
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_waypoint_result_callback)

    def cancel_move(self):
        if self._goal_handle is not None:
            self.get_logger().info('Attempting to cancel the move...')
            cancel_future = self._goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self.cancel_done_callback)
        else:
            self.get_logger().info('No active goal to cancel.')
    
    def cancel_done_callback(self, future):
        cancel_response = future.result()
        if len(cancel_response.goals_cancelled) > 0:
            self.get_logger().info('Goal cancellation accepted.')
            # self.destroy_node()
            # rclpy.shutdown()
            # sys.exit(0)
        else:
            self.get_logger().info('Goal cancellation failed or no active goal to cancel.')
    
    def get_waypoint_result_callback(self, future):
        result = future.result().result
        missed_waypoints = result.missed_waypoints
        if missed_waypoints:
            self.get_logger().info(f'Missed waypoints: {missed_waypoints}')
            # 웨이포인트를 놓친다면 다시 놓친 웨이 포인트 부터 시작

        else:
            self.get_logger().info('All waypoints completed successfully!')

            self.isHome = False
            self.active_patrol_mode()

    # zond에 대한 str을 받음
    def zone_callback(self, goal_handle):
        # goal 받았다는 로그
        goal = goal_handle.data
        if self.zone != goal and self.AMR_mode == 1:
            self.zone = goal
            self.active_patrol_mode()

    # 디지털 맵 내 지정 구역으로 이동
    def navigate_to_zone(self, zone_name):
        
        if zone_name in self.zones:
            # 목표 좌표 가져오기
            target_pose = self.zones[zone_name]

            # 목표 좌표를 PoseStamped 메시지로 생성
            goal_msg = PoseStamped()
            goal_msg.header.frame_id = 'map'  # SLAM에서 사용되는 좌표계 (보통 'map' 프레임)
            goal_msg.header.stamp = self.get_clock().now().to_msg()
            goal_msg.pose.position.x = target_pose.x
            goal_msg.pose.position.y = target_pose.y
            goal_msg.pose.orientation.w = 1.0  # 회전 값 (회전 없음)

            if not self.amr_navgoal_client.wait_for_server(timeout_sec=1.0):
                self.get_logger().info('Action server not available')
                return
            

            # 현재 amr 에 입력되어 있는 명령을 무시하도록 하는 코드 필요

            goal_pos = NavigateToPose.Goal()
            goal_pos.pose = goal_msg
            self.amr_navgoal_client.send_goal_async(goal_pos, feedback_callback=self.feedback_callback)

            # 목표를 move_base_simple/goal로 발행
            self.amr_goal_publisher_.publish(goal_msg)
            self.get_logger().info(f"Sending goal to zone {zone_name}: {target_pose}")

            

            # annotated_frame, track_ids, class_names, boxes, confidences = self.detect_objects()
            # if 'car' in class_names:
            #     self.tracking(self.frame, self.results)
            # else:
            #     move_cmd = Twist()
            #     radius = 5.0  # 원의 반지름
            #     angular_speed = 0.2  # 회전 속도
            #     linear_speed = angular_speed * radius  # 선형 속도

            #     move_cmd.linear.x = linear_speed
            #     move_cmd.angular.z = angular_speed

            #     # 일정 시간 동안 원을 그림
            #     rate = self.create_rate(10)
            #     start_time = self.get_clock().now()
            #     while (self.get_clock().now() - start_time) < Duration(seconds=2 * math.pi * radius / linear_speed):
            #         if 'car' in class_names:
            #             self.tracking(self.frame, self.results)
            #         else:
            #             self.amr_cmd_vel_pub.publish(move_cmd)
            #             rate.sleep()

            #     # 멈춤
            #     move_cmd.linear.x = 0.0
            #     move_cmd.angular.z = 0.0
            #     self.amr_cmd_vel_pub.publish(move_cmd)
        
    
    def tracking(self, frame):

        intruder_detection = None
        intruder_center_error = float('inf')  # 큰 값으로 초기화 
        screen_center_x = int(self.screen_width / 2)

        # Iterate over self.results to find the object with the highest confidence
        for result in self.results:
            if result['class_name'] == 'car':
                x1, y1, x2, y2 = result.boxes.data[:4]
                confidence = result['confidence']
                class_id = result['class_id']
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                center_error = abs(center_x - screen_center_x)  # change here

                if center_error < intruder_center_error: # change here
                    intruder_center_error = center_error
                    intruder_detection = (x1, y1, x2, y2, confidence, class_id)

        # If a detection is found, draw it and control the robot
        if intruder_detection:
            self.AMR_mode = 2 # tracking mode

            x1, y1, x2, y2, confidence, class_id = intruder_detection
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height

            # Draw the bounding box and center point on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            label_text = f'Conf: {confidence:.2f} Class: {int(class_id)}'
            cv2.putText(frame, label_text, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            

            # Compress the frame and convert it to a ROS 2 CompressedImage message
            _, compressed_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            compressed_img_msg = CompressedImage()
            compressed_img_msg.header.stamp = self.get_clock().now().to_msg()
            compressed_img_msg.format = "jpeg"
            compressed_img_msg.data = compressed_frame.tobytes()

            self.sm_tracked_image_publisher(compressed_img_msg)

  
            # Publish movement commands to align and move backward if the box area is too large
            twist = Twist()

            # Align the robot with the center of the screen
            alignment_tolerance = 20  # Pixel tolerance for alignment

            if abs(center_x - screen_center_x) > alignment_tolerance:
                if center_x < screen_center_x:
                    twist.angular.z = 0.2  # Rotate left
                else:
                    twist.angular.z = -0.2  # Rotate right
            else:
                twist.angular.z = 0.0  # Stop rotation when aligned

            # Move backward if the object's bounding box area is greater than or equal to 80% of the screen area
            if box_area < self.target_area_threshold:
                twist.linear.x = 0.1
            elif box_area > self.target_area_threshold:
                twist.linear.x = -0.1  # Move backward slowly
            else:
                twist.linear.x = 0.0  # Stop moving if the object is not too close

            self.amr_cmd_vel_pub.publish(twist)
        
        # Not found
        else:
            if self.AMR_mode == 2:
                self.AMR_mode = 1
                self.active_patrol_mode()


    def feedback_callback(self, feedback):
        # 네비게이션 피드백 처리 (필요시 사용)
        self.get_logger().info(f"Feedback: {feedback}")

    def handle_patrol_service_request(self,request,response):
        response.success = True
        if self.isHome:
            self.waypoint_goal()
        else: 
            self.active_patrol_mode()

        return response

    def active_patrol_mode(self):
        print("active patrol mode")
        if self.AMR_mode != 2:
            self.AMR_mode = 1
            self.navigate_to_zone(self.zone)
    
    def active_gohome_mode(self):
        if self.AMR_mode == 0:
            self.navigate_to_zone("Home")

    def handle_gohome_service_request(self, request, response):
        response.success = True

        self.AMR_mode = 0
        # goal_msg = PoseStamped()
        # goal_msg.header.frame_id = 'map'  # SLAM에서 사용되는 좌표계 (보통 'map' 프레임)
        # goal_msg.pose.position.x = self.home_pose.x
        # goal_msg.pose.position.y = self.home_pose.y
        # goal_msg.pose.orientation.w = 1.0  # 회전 값 (회전 없음)

        # # 목표를 move_base_simple/goal로 발행
        # self.amr_goal_publisher_.publish(goal_msg)
        # self.get_logger().info(f"Sending AMR to home")
        self.active_gohome_mode()

        return response
    
    def convert_cv2_to_ros_image(self, ros_image):
        # ROS2 이미지를 OpenCV로 변환
        bridge = CvBridge()
        return bridge.cv2_to_imgmsg(ros_image, encoding='bgr8')

    def image_callback(self):
        
        if self.AMR_mode == 0:
            return

        # 객체 감지 및 추적
        try:
            detections, track_ids, class_names, boxes, confidences = self.detect_objects(self.frame, self.self.results)
        except:
            return False
        
        # tracking
        self.tracking()

        ros_image = self.convert_cv2_to_ros_image(detections)
        # ros_image.header = msg.Header()
        # ros_image.header.stamp = self.get_clock().now().to_msg()
        self.AMR_image_publisher.publish(ros_image)
        self.get_logger().info("Publishing video frame")

    def amr_image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.amr_image_frame = frame

    def detect_objects(self):
        # Store the track history
        
        success, frame = self.cap.read()
        # YOLOv8 객체 탐지
        self.results = self.yolo_model.track(frame, persist=True)

        # Get the boxes and track IDs
        class_ids = self.results[0].boxes.cls
        class_names = [self.yolo_model.names[int(class_id)] for class_id in class_ids]
        boxes = self.results[0].boxes.xywh.cpu()
        track_ids = self.results[0].boxes.id.int().cpu().tolist()
        confidences = self.results[0].boxes.conf.cpu().tolist()
        # Visualize the self.results on the frame
        annotated_frame = self.results[0].plot()
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
    def publish_initial_pose(self):
        # Create the initial pose message
        initial_pose = PoseWithCovarianceStamped()
        initial_pose.header.frame_id = 'map'  # The frame in which the pose is defined
        initial_pose.header.stamp = self.get_clock().now().to_msg()

        # /initialpose

        # position:
        #   x: 0.1750425100326538
        #   y: 0.05808566138148308
        #   z: 0.0
        # orientation:
        #   x: 0.0
        #   y: 0.0
        #   z: -0.04688065682721989
        #   w: 0.9989004975549108
  


        # Set the position (adjust these values as needed)
        initial_pose.pose.pose.position.x = 0.043317 # X-coordinate
        initial_pose.pose.pose.position.y = 0.033049 # Y-coordinate
        initial_pose.pose.pose.position.z = 0.0  # Z should be 0 for 2D navigation

        # Set the orientation (in quaternion form)
        initial_pose.pose.pose.orientation = Quaternion(
            x=0.0,
            y=0.0,
            z=-0.04688065682721989,  # 90-degree rotation in yaw (example)
            w=0.9989004975549108  # Corresponding quaternion w component
        )

        # covariance:
        #   - 0.25
        #   - 0.0
        #   - 0.0
        #   - 0.0
        #   - 0.0
        #   - 0.0
        #   - 0.0
        #   - 0.25
        #   - 0.0
        #   - 0.0
        #   - 0.0
        #   - 0.0
        #   - 0.0
        #   - 0.0
        #   - 0.0
        #   - 0.0
        #   - 0.0
        #   - 0.0
        #   - 0.0
        #   - 0.0
        #   - 0.0
        #   - 0.0
        #   - 0.0
        #   - 0.0
        #   - 0.0
        #   - 0.0
        #   - 0.0
        #   - 0.0
        #   - 0.0
        #   - 0.0
        #   - 0.0
        #   - 0.0
        #   - 0.0
        #   - 0.0
        #   - 0.0
        #   - 0.06853891909122467

        # Set the covariance values for the pose estimation
        initial_pose.pose.covariance = [
            0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891909122467
        ]


        # Publish the initial pose
        self.amr_initialpose_publisher.publish(initial_pose)
        self.get_logger().info('Initial pose published.')

        # Destroy the node and shutdown rclpy
        # self.destroy_node()
        # rclpy.shutdown()
        # sys.exit(0)


def keyboard_listener(node):
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        while True:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1)
                if key.lower() == 's':
                    node.get_logger().info('Key "s" pressed. Cancelling goal...')
                    node.cancel_move()
                    break
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        

def main(args=None):
    rclpy.init(args=args)
    action_server = MoveToZoneActionServer()
    
    thread = threading.Thread(target=keyboard_listener, args=(action_server,), daemon=True)
    thread.start()

    rclpy.spin(action_server)  # 노드가 계속 실행되도록
    action_server.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

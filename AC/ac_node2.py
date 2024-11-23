import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import CompressedImage,Image
import std_msgs.msg as msg
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from ultralytics import YOLO
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import FollowWaypoints
from nav2_msgs.action import NavigateToPose
import math
import threading
import sys
import select
import termios
import tty
import cv2
###########################################
'''
Part: AC_tracking

수정사항

1. 많은 객체 중에 침입자를 결정 로직 재확인 필요 ( 고려중인 로직: 중앙값을 바라보고 터틀봇 프레임중앙과 가장 가까운 박스 중앙값을 갖는 객체를 침입자로)
2. area에 대응하는 디지털 맵 내의 좌표값 얻기(AMR 카메라를 통해 해당 영역을 가장 잘 관찰할 수 있으면서 해당 영역의 중앙값을 바라보도록 설계)
3. qos 설정 여부 결정


'''
###########################################

class AMRControlNode():
    def __init__(self):
        super().__init__('amr_control_node')

        # AMR_waypoint_action_client
        self.amr_waypoint_action_client = ActionClient(self, FollowWaypoints, '/follow_waypoints')
        # AMR_NavGoal_action_client
        self.amr_navgoal_action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        # AMR_cmd_topic_pub
        self.amr_cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        # SM_tracked_image_topic_pub
        self.sm_tracked_image_publisher = self.create_publisher(Image, 'tracked_image', 10)
        # DA_area_sub
        self.da_area_subscription  = self.create_subscription(msg.String,'navigate_to_zone',self.get_area_callback, qos_profile )

        self.bridge = CvBridge()
        self.model = YOLO("./yolov8n.pt")
        self.cap = cv2.VideoCapture(4)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        self.timer = self.create_timer(0.1, self.tracking_timer_callback) 

        # Get screen dimensions and calculate 80% threshold for bounding box area
        self.screen_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.screen_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.screen_area = self.screen_width * self.screen_height
        self.target_area_threshold = 0.8 * self.screen_area

        # area값에 대응되는 디지털 맵 내 좌표(일요일에 수정)
        self.area_name = None
        self.areas = {
            'A': Point(x=1.0, y=2.0),
            'B': Point(x=3.0, y=4.0),
            'C': Point(x=5.0, y=6.0),
            'D': Point(x=7.0, y=8.0),
            'E': Point(x=9.0, y=10.0),
            'F': Point(x=11.0, y=12.0)
        }

    def euler_to_quaternion(self, roll, pitch, yaw):
        # Convert Euler angles to a quaternion
        qx = math.sin(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) - math.cos(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
        qy = math.cos(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2)
        qz = math.cos(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2) - math.sin(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2)
        qw = math.cos(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
        return Quaternion(x=qx, y=qy, z=qz, w=qw)
    
    def tracking_timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to capture frame from webcam.")
            return

        # Run tracking and get results
        results = self.model.track(source=frame, show=False, tracker='bytetrack.yaml')

        intruder_detection = None
        intruder_center_error = 0.0 

        # Iterate over results to find the object with the highest confidence
        for result in results:
            if result['class_id'] == 'car':
                x1, y1, x2, y2 = result.boxes.data[:4]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                center_error = abs(center_x - screen_center_x)  # change here

                if center_error < intruder_center_error: # change here
                    intruder_center_error = center_error
                    intruder_detection = (x1, y1, x2, y2, confidence, class_id)

        # If a detection is found, draw it and control the robot
        if intruder_detection:
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
            screen_center_x = int(self.screen_width / 2)
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

            self.amr_cmd_publisher.publish(twist)
        

    def waypoint_goal(self):
        # 세 개의 웨이포인트 정의 
        waypoints = []

        # 첫 번째 웨이포인트
        waypoint1 = PoseStamped()
        waypoint1.header.stamp.sec = 0
        waypoint1.header.stamp.nanosec = 0
        waypoint1.header.frame_id = "map"  # 프레임 ID를 설정 (예: "map")
        waypoint1.pose.position.x = 0.35624730587005615
        waypoint1.pose.position.y = -0.7531262636184692
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
        waypoint2.pose.position.x = -1.0062505006790161
        waypoint2.pose.position.y = -0.15937140583992004
        waypoint2.pose.position.z = 0.0
        
        waypoint2_yaw = 0.0  # Target orientation in radians
        waypoint2.pose.orientation = self.euler_to_quaternion(0, 0, waypoint2_yaw)
        
        # waypoint2.pose.orientation.x = 0.0
        # waypoint2.pose.orientation.y = 0.0
        # waypoint2.pose.orientation.z = -0.9999330665398213
        # waypoint2.pose.orientation.w = 0.01156989370173046
        waypoints.append(waypoint2)

        # 세 번째 웨이포인트
        waypoint3 = PoseStamped()
        waypoint3.header.stamp.sec = 0
        waypoint3.header.stamp.nanosec = 0
        waypoint3.header.frame_id = "map"  # 프레임 ID를 설정 (예: "map")
        waypoint3.pose.position.x = -1.443751335144043
        waypoint3.pose.position.y = -0.3468696177005768
        waypoint3.pose.position.z = 0.0
        
        waypoint3_yaw = 0.0  # Target orientation in radians
        waypoint3.pose.orientation = self.euler_to_quaternion(0, 0, waypoint3_yaw)
                
        # waypoint3.pose.orientation.x = 0.0
        # waypoint3.pose.orientation.y = 0.0
        # waypoint3.pose.orientation.z = -0.6938991006274311
        # waypoint3.pose.orientation.w = 0.7200722450896453
        waypoints.append(waypoint3)

        # FollowWaypoints 액션 목표 생성 및 전송
        goal_msg = FollowWaypoints.Goal()
        goal_msg.poses = waypoints

        # 서버 연결 대기
        self.amr_waypoint_action_client.wait_for_server()

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
        else:
            self.get_logger().info('All waypoints completed successfully!')

    def get_area_callback(self):
        self.area_name= msg.data
    
    def Nav_goal(self):
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        
        # Set goal position (update these values as needed)
        goal_msg.pose.pose.position.x = self.areas[self.area_name].x  # Area X coordinate
        goal_msg.pose.pose.position.y = self.areas[self.area_name].y # Area Y coordinate
        goal_msg.pose.pose.position.z = 0.0  # Z is typically 0 for 2D navigation

        # Set goal orientation using the euler_to_quaternion method
        goal_yaw = 0.0  # Target orientation in radians
        goal_msg.pose.pose.orientation = self.euler_to_quaternion(0, 0, goal_yaw)

        # Wait for the action server to be available
        self.amr_navgoal_action_client.wait_for_server()
        self.get_logger().info('Sending goal...')
        
        # Send the goal asynchronously and set up a callback for response
        self._send_goal_future = self.amr_navgoal_action_client.send_goal_async(goal_msg, feedback_callback=self.Nav_feedback_callback)
        self._send_goal_future.add_done_callback(self.Nav_goal_response_callback)
    
    def Nav_feedback_callback(self, feedback_msg):
        self.get_logger().info(f'Current position: {feedback_msg.feedback.current_pose.pose}')

    def Nav_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected.')
            return

        self.get_logger().info('Goal accepted.')
        self._goal_handle = goal_handle 
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_Nav_result_callback)
        
    def get_Nav_result_callback(self,future):
        result = future.result().result
        navigating_goal = result.navigating_goal # 진행중이던 goal 좌표가 있다면
        if navigating_goal:
            self.get_logger().info(f'A goal in progress: {navigating_goal}')
        else:
            self.get_logger().info('Reached the goal')
            
    

def keyboard_listener(node):
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        while True:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1)
                if key.lower() == 'n':
                    node.get_logger().info('Key "g" pressed. Sending goal...')
                    node.Nav_goal()
                elif key.lower() == 'w':
                    node.get_logger().info('Key "w" pressed. Sending waypoint')
                elif key.lower() == 's':
                    node.get_logger().info('Key "s" pressed. Cancelling goal...')
                    node.cancel_move()
                    break
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def destroy_node(self):
        super().destroy_node()
        self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    node = AMRControlNode()

    thread = threading.Thread(target=keyboard_listener, args=(node,), daemon=True)  # 동시에 키 리쓰너 시작
    thread.start()

    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
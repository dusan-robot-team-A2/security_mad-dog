import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
import std_msgs.msg as msg
from sensor_msgs.msg import Image
import time
import cv2
from cv_bridge import CvBridge
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy

qos_profile = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
    durability=QoSDurabilityPolicy.VOLATILE)


class MoveToPositionServer(Node):
    def __init__(self):
        super().__init__('move_to_position_server')
        self.publisher_alter = self.create_publisher(msg.String, 'topic', qos_profile=qos_profile)
        
        self.publisher_cam = self.create_publisher(Image, 'topic', qos_profile=qos_profile)

        # 목표 위치를 계속 업데이트하면서 이동하기 위해 use_timer
        self.alter_publish_once()

        self.video_timer = self.create_timer(1.0, self.publish_video)
        self.bridge = CvBridge()
        self.cam = cv2.VideoCapture(0)

    def publish_video(self):
        # 이미지 파일을 읽음

        ret, frame = self.cam.read()
        
        try:
            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting OpenCV image to ROS Image message: {e}")
            return
        
        # 퍼블리시: Image 메시지 전송
        ros_image.header = msg.Header()
        ros_image.header.stamp = self.get_clock().now().to_msg()  # 타임스탬프 설정
        self.publisher_cam.publish(ros_image)
        self.get_logger().info("Publishing video frame")

    def alter_publish_once(self):
        # 퍼블리시할 메시지 생성
        message = msg.String()
        message.data = 'Test'

        # 메시지를 퍼블리시
        self.publisher_alter.publish(message)
        self.get_logger().info(f'Publishing: {message.data}')


def main(args=None):
    rclpy.init(args=args)
    alter = MoveToPositionServer()
    rclpy.spin(alter)
    alter.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

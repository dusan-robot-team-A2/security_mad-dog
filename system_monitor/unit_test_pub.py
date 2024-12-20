import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from mad_dog_interface.srv import ActivatePatrol
from mad_dog_interface.srv import ActivateGoHome
import random

class SimplePublisher(Node):
    def __init__(self):
        super().__init__('simple_publisher')  # 노드 이름
        self.da_alert_pub = self.create_publisher(Bool, 'da_alert', 10)
        self.amr_alert_pub = self.create_publisher(Bool, 'amr_alert', 10)
        self.amr_position_pub = self.create_publisher(Point, 'amr_position', 10)

        self.patrol_service = self.create_service(ActivatePatrol, 'patrol_service', self.handle_patrol_service_request)
        self.gohome_service = self.create_service(ActivateGoHome, 'gohome_service', self.handle_gohome_service_request)

        self.timer = self.create_timer(1.0, self.timer_callback)  # 1초마다 콜백 실행
        self.get_logger().info('Publisher node has been started.')

    def timer_callback(self):
        da_alert = Bool()
        da_alert.data = bool(random.randint(0,1))
        self.da_alert_pub.publish(da_alert)
        self.get_logger().info(f'DA Alert Pub: "{da_alert.data}"') 

        amr_alert = Bool()
        amr_alert.data = bool(random.randint(0,1))
        self.amr_alert_pub.publish(amr_alert)
        self.get_logger().info(f'AMR Alert Pub: "{amr_alert.data}"') 

        amr_position = Point()
        amr_position.x = float(random.randint(0,300))
        amr_position.y = float(random.randint(0,300))
        self.amr_position_pub.publish(amr_position)
        self.get_logger().info(f'AMR pos Pub: "{str(amr_position)}"') 
    
    def handle_patrol_service_request(self, request, response):
        response.success = True
        print("patrol mod")
        return response

    def handle_gohome_service_request(self, request, response):
        response.success = True
        print("gohome mod")
        return response


def main(args=None):
    rclpy.init(args=args)  # ROS 2 초기화
    node = SimplePublisher()  # 퍼블리셔 노드 생성

    try:
        rclpy.spin(node)  # 노드 실행
    except KeyboardInterrupt:
        pass  # Ctrl+C로 종료
    finally:
        node.destroy_node()  # 노드 소멸
        rclpy.shutdown()  # ROS 2 종료

if __name__ == '__main__':
    main()

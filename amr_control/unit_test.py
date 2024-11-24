import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from mad_dog_interface.srv import ActivatePatrol
from mad_dog_interface.srv import ActivateGoHome
from nav2_msgs.action import NavigateToPose, FollowWaypoints
import time
import random

class SimplePublisher(Node):
    def __init__(self):
        super().__init__('simple_publisher')  # 노드 이름
        self.zone_pub = self.create_publisher(String, 'navigate_to_zone', 10)
        self.amr_waypoint_action_server = ActionServer(
            self,
            FollowWaypoints,
            '/follow_waypoints',  # Action 이름
            self.amr_waypoint_execute_callback
        )
        self.amr_navgoal_action_server = ActionServer(
            self,
            NavigateToPose,
            '/navigate_to_pose',  # Action 이름
            self.amr_navgoal_execute_callback
        )

        self.timer = self.create_timer(10.0, self.zone_callback)  # 1초마다 콜백 실행

    def zone_callback(self):
        msg = String()
        msg.data = "A"
        self.zone_pub.publish(msg)
        
    
    def amr_waypoint_execute_callback(self, goal_handle):
        self.get_logger().info('Way point')
        result = FollowWaypoints.Result()
        feedback_msg = FollowWaypoints.Feedback()

        time.sleep(5)

        result.sequence = True
        goal_handle.succeed()

        return result
    
    def amr_navgoal_execute_callback(self, goal_handle):
        self.get_logger().info('Nav Goal')
        result = NavigateToPose.Result()
        feedback_msg = NavigateToPose.Feedback()

        result.sequence = True
        goal_handle.succeed()

        return result


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

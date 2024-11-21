import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

class DataSubscriber(Node):
    def __init__(self):
        super().__init__('data_subscriber')
        self.subscription = self.create_subscription(
            String,
            'detection',
            self.data_callback,
            10
        )
        self.subscription  # Prevent unused variable warning
        self.get_logger().info('Data Subscriber Node has been started.')

    def data_callback(self, msg):
        msg_dic = json.load(msg.data)
        # Parse the received message, expecting the format: "id,timestamp"
        for k, v in msg_dic.items():
            self.get_logger().info(f'id: {k}, box: {v[0]["box"]}, confidence: {v[0]["confidence"]}')
        
def main(args=None):
    rclpy.init(args=args)
    node = DataSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
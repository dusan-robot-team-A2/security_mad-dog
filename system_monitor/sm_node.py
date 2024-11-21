import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
import json


class SystemMonitoringNode(Node):
    def __init__(self):
        super().__init__('data_subscriber')

        #DA cctv Image
        self.cctv_image_subscribtion = self.create_subscription(
            Image,
            'cctv_image', 
            self.cctv_image_callback,
            10
        )

        #DA Track Alert
        self.da_alert_subscription = self.create_subscription(
            Bool,
            'da_alert',
            self.da_alert_callback,
            10
        )
        #DA yolo tracking data
        self.da_track_data_subscription = self.create_subscription(
            String,
            'da_track_data',
            self.da_track_data_callback,
            10
        )

        #AMR Alert
        self.amr_alert_subscription = self.create_subscription(
            Bool,
            'amr_alert',
            self.amr_alert_callback,
            10
        )
        #AMR Pos
        self.amr_image_subscription = self.create_subscription(
            Point,
            'amr_position',
            self.amr_position_callback,
            10
        )
        #AMR Image
        self.amr_image_subscription = self.create_subscription(
            Image,
            'amr_image',
            self.amr_image_callback,
            10
        )

        # Prevent unused variable warning
        self.cctv_image_subscribtion
        self.da_alert_subscription
        self.da_track_data_subscription
        self.amr_alert_subscription
        self.amr_image_subscription


        # System variables
        self.bridge = CvBridge()

        # Global variables of SM
        self.emergency_status = False
        self.amr_status = 0 # 0: normal,1: patrol, 2: follow

        self.cctv_image_frame = None
        self.da_alert = False
        self.da_track_data = []

        self.amr_alert = False
        self.amr_position = None
        self.amr_image_frame = None

    def cctv_image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.cctv_image_frame = frame

    def da_alert_callback(self, msg):
        self.da_alert = msg.data
        if not self.emergency_status and self.da_alert: #emergency 최초 발생 situation
            self.emergency_status = True
            #ac 의 patrol service 요청과 관련된 로직 필요
            print("ac의 patrol service 요청") 
        

    def da_track_data_callback(self, msg):
        json_data = msg.data
        data = json.loads(json_data)

        self.da_track_data = data

    def amr_alert_callback(self, msg):
        self.amr_alert = msg.data

        # emergency management
        if self.emergency_status:
            if self.amr_alert: # emergency상황에서 amr 이 물체를 detect -> follow
                self.amr_status = 2
            else: # emergency상황에서 amr 이 물체를 detectX -> patrol
                self.amr_status = 1
    
    def amr_position_callback(self,msg):
        self.amr_position = msg

    def amr_image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.amr_image_frame = frame

    # emergency종료
    def end_emergency(self):
        self.emergency_status = False
        self.amr_status = 0
        #ac의 go_home service 호출
        print("ac의 go_home service 요청")

    

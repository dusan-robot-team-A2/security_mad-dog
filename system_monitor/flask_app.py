from flask import Flask, Response, render_template
from sm_node import SystemMonitoringNode
from threading import Thread
import cv2
import rclpy
import time

rclpy.init()
smNode = SystemMonitoringNode()
app = Flask(__name__)

# ROS 2 노드를 별도의 스레드에서 실행
def spin_ros2_node(node):
    rclpy.spin(node)
    rclpy.shutdown()
        
def generate_frames_cctv():
    while True:
        # Read the camera frame
        time.sleep(0.1)
        frame = smNode.cctv_image_frame

        if frame is not None:
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # Concatenate frame bytes with multipart data structure
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

# socket으로 데이터 전송하도록 변경 필요
@app.route('/data')
def data():
    result = f"<p>Emergency Status: {smNode.emergency_status}\n</p>"
    result += f"<p>AMR Status: {smNode.amr_status}\n</p>"
    result += f"<p>AMR Position: {smNode.amr_position}\n</p>"
    
    return result

@app.route('/video_feed')
def video_feed():
    # Returns the video stream response
    return Response(generate_frames_cctv(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    ros2_thread = Thread(target=spin_ros2_node, args=[smNode], daemon=True)
    ros2_thread.start()
    app.run(debug=True)
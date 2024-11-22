from flask import Flask, Response, render_template, Request
from sm_node import SystemMonitoringNode
from threading import Thread
import cv2
import rclpy
import time
import json

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
            
def generate_sm_variable():
    while True:
        time.sleep(0.1)

        amr_position = smNode.amr_position

        data = {
            "emergency_status": smNode.emergency_status,
            "amr_status": smNode.amr_status,
            "amr_positioin": [amr_position.x, amr_position.y] if amr_position else None,
            "da_track_data": smNode.da_track_data,
        }

        yield f"data: {json.dumps(data)}\n\n"

@app.route('/')
def index():
    return render_template('index.html')

# socket으로 데이터 전송하도록 변경 필요
@app.route('/data')
def data():
    return Response(generate_sm_variable(), content_type='text/event-stream')

@app.route('/video_feed')
def video_feed():
    # Returns the video stream response
    return Response(generate_frames_cctv(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/end_emergency', methods=['POST'])
def end_emergency():
    print("end emergency")
    smNode.end_emergency()
    return Response(status=200)
    

if __name__ == "__main__":
    ros2_thread = Thread(target=spin_ros2_node, args=[smNode], daemon=True)
    ros2_thread.start()
    app.run(debug=True)
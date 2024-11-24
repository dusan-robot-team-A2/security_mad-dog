import cv2
import numpy as np

# 글로벌 변수 초기화
points = []  # 현재 클릭한 점들
polygons = []  # 완성된 다각형 리스트
is_polygon_complete = False  # 현재 다각형이 완성되었는지 확인


def mouse_callback(event, x, y, flags, param):
    """마우스 클릭 이벤트 콜백 함수"""
    global points, is_polygon_complete, polygons

    if event == cv2.EVENT_LBUTTONDOWN and not is_polygon_complete:
        # 좌표 추가
        points.append((x, y))
        print(f"Point added: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN and not is_polygon_complete:
        # 우클릭으로 다각형 완성
        if len(points) > 2:
            is_polygon_complete = True
            polygons.append(points.copy())  # 완성된 다각형 저장
            print("Polygon completed!")
            print("Polygon points:", points)
            points = []  # 새로운 다각형을 시작하기 위해 현재 점 초기화
            is_polygon_complete = False
        else:
            print("At least 3 points are required to form a polygon.")


# 비디오 캡처 설정
cap = cv2.VideoCapture(0)  # 0번 카메라는 웹캠, 비디오 파일은 경로를 입력
cv2.namedWindow("Video")
cv2.setMouseCallback("Video", mouse_callback)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video.")
        break

    # 이전에 만든 모든 다각형 그리기
    for polygon in polygons:
        pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # 현재 클릭 중인 점 및 다각형 표시
    for point in points:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)  # 빨간 점 표시
    if len(points) > 2:
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=False, color=(255, 255, 0), thickness=1)

    # 화면에 출력
    cv2.imshow("Video", frame)

    # ESC 키로 종료
    key = cv2.waitKey(1)
    if key == 27:  # ESC 키
        break
    elif key == ord('r'):  # 'r' 키로 모든 다각형 초기화
        points = []
        polygons = []
        print("Reset all polygons.")

cap.release()
cv2.destroyAllWindows()

import cv2

# 카메라 장치 열기 (0은 기본 카메라를 의미)
cap = cv2.VideoCapture(0)

# 카메라 열기 실패 여부 확인
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

while True:
    # 프레임 캡처
    ret, frame = cap.read()

    # 프레임 캡처 성공 여부 확인
    if not ret:
        print("프레임을 가져오는 데 실패했습니다.")
        break

    # 캡처된 프레임을 윈도우에 표시
    cv2.imshow('Camera Feed', frame)

    # 'q' 키를 눌러 창을 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라 장치 해제
cap.release()

# 열린 모든 창을 닫기
cv2.destroyAllWindows()

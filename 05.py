import cv2
import mediapipe as mp
import numpy as np
import time
import random
import math
from collections import deque
from pathlib import Path

# =====================================================
# 1. 기본 설정
# =====================================================

WIDTH = 1280
HEIGHT = 720
FPS = 30

MODEL_PATH = "src/models/MediaPipe/hand_landmarker.task"

# 공 관련 설정
BALL_RADIUS = 45
MIN_RADIUS = 15
MAX_RADIUS = 145
POP_RADIUS = 115

# 물리 설정 (단위: 픽셀, 초)
GRAVITY = 1100.0
BOUNCE = 0.78
FRICTION = 0.995
MAX_THROW_SPEED = 1800.0

# 제스처 설정
GRAB_THRESHOLD = 0.32
RELEASE_THRESHOLD = 0.55

# MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# =====================================================
# 2. Jetson GStreamer 카메라 파이프라인
# =====================================================

pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), "
    "width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(
    pipeline, cv2.CAP_GSTREAMER
)

if not cap.isOpened():
    raise RuntimeError("카메라 연결 실패")

if not Path(MODEL_PATH).is_file():
    cap.release()
    raise FileNotFoundError(MODEL_PATH)

# =====================================================
# 3. 가상 공 클래스
# =====================================================

class Ball:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = WIDTH / 2
        self.y = HEIGHT / 2
        self.vx = 0.0
        self.vy = 0.0
        self.radius = BALL_RADIUS
        self.state = "FREE"

    def physics(self, dt):
        if self.state != "FREE":
            return

        # 중력
        self.vy += GRAVITY * dt

        # 위치 갱신
        self.x += self.vx * dt
        self.y += self.vy * dt

        # 공기 저항
        damping = FRICTION ** (dt * FPS)
        self.vx *= damping

        r = self.radius

        # 왼쪽 벽
        if self.x - r < 0:
            self.x = r
            self.vx = abs(self.vx) * BOUNCE

        # 오른쪽 벽
        if self.x + r > WIDTH:
            self.x = WIDTH - r
            self.vx = -abs(self.vx) * BOUNCE

        # 위쪽 벽
        if self.y - r < 0:
            self.y = r
            self.vy = abs(self.vy) * BOUNCE

        # 바닥
        if self.y + r > HEIGHT:
            self.y = HEIGHT - r
            self.vy = -abs(self.vy) * BOUNCE

            # 작은 진동 제거
            if abs(self.vy) < 50:
                self.vy = 0

            self.vx *= 0.96


# =====================================================
# 4. 파티클 클래스 (공 터뜨리기)
# =====================================================

class Particle:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(150, 650)

        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed

        self.life = random.uniform(0.5, 1.2)
        self.max_life = self.life

        self.radius = random.randint(3, 9)

        self.color = random.choice([
            (255, 255, 0),
            (255, 100, 0),
            (255, 0, 255),
            (0, 255, 255),
            (0, 150, 255),
        ])

    def update(self, dt):
        self.vy += GRAVITY * 0.4 * dt

        self.x += self.vx * dt
        self.y += self.vy * dt

        self.life -= dt

    def draw(self, frame):
        if self.life <= 0:
            return

        size = max(
            1,
            int(self.radius * self.life / self.max_life)
        )

        cv2.circle(
            frame,
            (int(self.x), int(self.y)),
            size,
            self.color,
            -1
        )


# =====================================================
# 5. MediaPipe 손 좌표 추출
# =====================================================

def get_hands(result, width, height):
    hands = []

    for landmarks in result.hand_landmarks:

        points = []

        for landmark in landmarks:
            points.append((
                int(landmark.x * width),
                int(landmark.y * height)
            ))

        # 엄지 끝: 4
        # 검지 끝: 8
        thumb = np.array(points[4], dtype=np.float32)
        index = np.array(points[8], dtype=np.float32)

        # 두 손가락 중간 지점
        pinch_center = (thumb + index) / 2

        # 엄지-검지 거리
        pinch_distance = np.linalg.norm(
            thumb - index
        )

        # 손바닥 크기를 이용한 정규화
        palm_size = np.linalg.norm(
            np.array(points[0], dtype=np.float32)
            - np.array(points[9], dtype=np.float32)
        )

        palm_size = max(palm_size, 1.0)

        pinch_ratio = pinch_distance / palm_size

        hands.append({
            "points": points,
            "center": pinch_center,
            "pinch_distance": pinch_distance,
            "pinch_ratio": pinch_ratio,
        })

    return hands


# =====================================================
# 6. Kalman Filter - 손 위치 안정화
# =====================================================

kalman = cv2.KalmanFilter(4, 2)

kalman.transitionMatrix = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
], dtype=np.float32)

kalman.measurementMatrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
], dtype=np.float32)

kalman.processNoiseCov = np.eye(
    4, dtype=np.float32
) * 0.05

kalman.measurementNoiseCov = np.eye(
    2, dtype=np.float32
) * 2.0

kalman.errorCovPost = np.eye(
    4, dtype=np.float32
)


def initialize_kalman(x, y):
    state = np.array(
        [[x], [y], [0], [0]],
        dtype=np.float32
    )

    kalman.statePre = state.copy()
    kalman.statePost = state.copy()

    kalman.errorCovPost = np.eye(
        4, dtype=np.float32
    )


def update_kalman(x, y):
    kalman.predict()

    z = np.array(
        [[x], [y]],
        dtype=np.float32
    )

    corrected = kalman.correct(z)

    return (
        float(corrected[0, 0]),
        float(corrected[1, 0])
    )


# =====================================================
# 7. 공 및 제스처 상태
# =====================================================

ball = Ball()
particles = []

# 공을 잡은 손의 최근 위치
hand_history = deque(maxlen=12)

# 잡은 손 식별용 마지막 위치
grab_hand_position = None

# Resize 기준값
resize_base_distance = None
resize_base_radius = None

# 손 추적이 일시적으로 사라진 시각
hand_lost_since = None

# 터진 이후 재생성 시각
pop_time = None

# 공이 터졌는지 여부
popped = False


def reset_interaction():
    global grab_hand_position
    global resize_base_distance
    global resize_base_radius
    global hand_lost_since

    grab_hand_position = None
    resize_base_distance = None
    resize_base_radius = None
    hand_lost_since = None
    hand_history.clear()


def throw_ball():
    # 최근 손의 움직임으로 던지는 속도 계산
    ball.vx = 0.0
    ball.vy = 0.0

    if len(hand_history) >= 2:
        t1, x1, y1 = hand_history[0]
        t2, x2, y2 = hand_history[-1]

        dt = max(t2 - t1, 1e-3)

        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt

        # 지나치게 큰 속도 제한
        speed = math.hypot(vx, vy)

        if speed > MAX_THROW_SPEED:
            scale = MAX_THROW_SPEED / speed
            vx *= scale
            vy *= scale

        ball.vx = vx
        ball.vy = vy

    ball.state = "FREE"
    reset_interaction()


# =====================================================
# 8. 실시간 실행
# =====================================================

start_time = time.monotonic()
prev_time = start_time
last_timestamp_ms = -1

try:
    with HandLandmarker.create_from_options(options) as detector:

        while True:

            ret, frame = cap.read()

            if not ret:
                print("프레임 읽기 실패")
                break

            # 거울 모드
            frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]

            now = time.monotonic()
            dt = min(now - prev_time, 0.05)
            prev_time = now

            # -------------------------------------
            # MediaPipe 손 검출
            # -------------------------------------

            rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb)
            )

            timestamp_ms = int(
                (now - start_time) * 1000
            )

            timestamp_ms = max(
                timestamp_ms,
                last_timestamp_ms + 1
            )
            last_timestamp_ms = timestamp_ms

            result = detector.detect_for_video(
                mp_image,
                timestamp_ms
            )

            hands = get_hands(result, w, h)

            # -------------------------------------
            # 공이 터졌을 경우
            # -------------------------------------

            if popped:
                for particle in particles:
                    particle.update(dt)

                particles = [
                    p for p in particles if p.life > 0
                ]

                # 1.5초 후 새로운 공 생성
                if now - pop_time > 1.5:
                    ball.reset()
                    reset_interaction()
                    popped = False
                    particles.clear()

            # -------------------------------------
            # 공을 잡고 있지 않은 상태
            # -------------------------------------

            if not popped and ball.state == "FREE":

                ball.physics(dt)

                # 손 중에서 공 근처를 핀치한 손 선택
                for hand in hands:

                    pos = hand["center"]
                    distance = math.hypot(
                        pos[0] - ball.x,
                        pos[1] - ball.y
                    )

                    if (
                        hand["pinch_ratio"] < GRAB_THRESHOLD
                        and distance < ball.radius + 45
                    ):
                        ball.state = "GRABBED"

                        grab_hand_position = pos.copy()

                        initialize_kalman(
                            float(pos[0]),
                            float(pos[1])
                        )

                        ball.vx = 0.0
                        ball.vy = 0.0

                        hand_history.clear()
                        hand_history.append((
                            now, float(pos[0]), float(pos[1])
                        ))

                        resize_base_distance = None
                        resize_base_radius = None

                        break

            # -------------------------------------
            # 공을 잡고 있는 상태
            # -------------------------------------

            elif not popped and ball.state == "GRABBED":

                # 이전 위치와 가장 가까운 손을
                # 잡고 있는 손으로 선택
                grabbing_hand = None
                other_hand = None

                if len(hands) > 0:

                    distances = [
                        np.linalg.norm(
                            hand["center"] - grab_hand_position
                        )
                        for hand in hands
                    ]

                    idx = int(np.argmin(distances))

                    # 갑자기 다른 손으로 바뀌는 현상 방지
                    if distances[idx] < 180:
                        grabbing_hand = hands[idx]

                        if len(hands) > 1:
                            other_hand = hands[1 - idx]

                if grabbing_hand is not None:

                    hand_lost_since = None

                    pos = grabbing_hand["center"]

                    grab_hand_position = pos.copy()

                    # --------------------------------
                    # 손가락 핀치를 풀면 던지기
                    # --------------------------------

                    if grabbing_hand["pinch_ratio"] > RELEASE_THRESHOLD:

                        throw_ball()

                    else:
                        # Kalman Filter로 손 위치 보정
                        x_bel, y_bel = update_kalman(
                            float(pos[0]),
                            float(pos[1])
                        )

                        ball.x = x_bel
                        ball.y = y_bel

                        hand_history.append((
                            now, x_bel, y_bel
                        ))

                        # 최근 0.15초의 손 궤적 유지
                        while (
                            len(hand_history) > 2
                            and now - hand_history[0][0] > 0.15
                        ):
                            hand_history.popleft()

                        # ----------------------------
                        # 다른 손으로 공 크기 조절
                        # ----------------------------

                        if other_hand is not None:

                            d = other_hand["pinch_ratio"]

                            # 처음 다른 손이 등장하면 기준 저장
                            if resize_base_distance is None:

                                resize_base_distance = d
                                resize_base_radius = ball.radius

                            else:
                                change = (
                                    d - resize_base_distance
                                )

                                ball.radius = (
                                    resize_base_radius
                                    + change * 130
                                )

                                ball.radius = max(
                                    MIN_RADIUS,
                                    min(MAX_RADIUS, ball.radius)
                                )

                        else:
                            resize_base_distance = None
                            resize_base_radius = None

                        # ----------------------------
                        # 공 터뜨리기
                        # ----------------------------

                        if ball.radius >= POP_RADIUS:

                            popped = True
                            pop_time = now

                            particles = [
                                Particle(ball.x, ball.y)
                                for _ in range(90)
                            ]

                            ball.state = "POPPED"
                            reset_interaction()

                else:
                    # 손을 잠깐 놓쳐도 바로 던지지 않음
                    if hand_lost_since is None:
                        hand_lost_since = now

                    if now - hand_lost_since > 0.2:
                        # 추적이 사라진 경우에는
                        # 무작위 속도로 던지지 않음
                        ball.vx = 0.0
                        ball.vy = 0.0
                        ball.state = "FREE"
                        reset_interaction()

            # -------------------------------------
            # 9. 손 랜드마크 표시
            # -------------------------------------

            for hand in hands:

                points = hand["points"]

                # 엄지 끝
                cv2.circle(
                    frame, points[4], 7,
                    (0, 255, 255), -1
                )

                # 검지 끝
                cv2.circle(
                    frame, points[8], 7,
                    (255, 0, 255), -1
                )

                # 손가락 사이 연결선
                cv2.line(
                    frame,
                    points[4],
                    points[8],
                    (255, 255, 255),
                    2
                )

            # -------------------------------------
            # 10. 가상 공 그리기
            # -------------------------------------

            if not popped:

                center = (
                    int(ball.x),
                    int(ball.y)
                )

                radius = int(ball.radius)

                # 반투명 공
                overlay = frame.copy()

                cv2.circle(
                    overlay,
                    center,
                    radius,
                    (255, 150, 30),
                    -1
                )

                frame = cv2.addWeighted(
                    overlay,
                    0.55,
                    frame,
                    0.45,
                    0
                )

                # 공의 외곽선
                cv2.circle(
                    frame,
                    center,
                    radius,
                    (255, 255, 255),
                    3
                )

                # 중심 빛 효과
                cv2.circle(
                    frame,
                    center,
                    max(3, radius // 4),
                    (255, 255, 200),
                    -1
                )

            # 파티클 그리기
            for particle in particles:
                particle.draw(frame)

            # -------------------------------------
            # 11. UI 정보
            # -------------------------------------

            state_text = "POPPED" if popped else ball.state

            cv2.putText(
                frame,
                f"STATE: {state_text}",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"RADIUS: {int(ball.radius)}",
                (30, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "PINCH: GRAB | OTHER HAND: RESIZE",
                (30, h - 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "RELEASE: THROW | R: RESET | Q: QUIT",
                (30, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.imshow("AirBall AR", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                ball.reset()
                particles.clear()
                popped = False
                reset_interaction()

finally:
    cap.release()
    cv2.destroyAllWindows()

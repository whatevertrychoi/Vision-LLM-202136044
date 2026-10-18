import cv2
import mediapipe as mp
import numpy as np
import time
import math
from collections import deque
from pathlib import Path

# AirBall Football: hand throw + foot/knee juggling, 2D webcam prototype.
WIDTH, HEIGHT = 640, 480
HAND_MODEL = Path('src/models/MediaPipe/hand_landmarker.task')
POSE_MODEL = Path('src/models/MediaPipe/pose_landmarker_full.task')

pipeline = (
    'v4l2src device=/dev/video0 ! '
    'video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! '
    'videoconvert ! video/x-raw,format=BGR ! '
    'queue leaky=downstream max-size-buffers=1 '
    'max-size-bytes=0 max-size-time=0 ! '
    'appsink drop=true max-buffers=1 sync=false'
)

for file in (HAND_MODEL, POSE_MODEL):
    if not file.is_file():
        raise FileNotFoundError(f'Model not found: {file}. Download it as instructed.')

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    raise RuntimeError('Camera failed to open. Check /dev/video0 and GStreamer pipeline.')

BaseOptions = mp.tasks.BaseOptions
Vision = mp.tasks.vision
mode = Vision.RunningMode.VIDEO
hand_options = Vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
    running_mode=mode, num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5)
pose_options = Vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
    running_mode=mode, num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5)

GRAVITY = 790.0  # px/s^2
BOUNCE = 0.35
BALL_RADIUS = 18
PINCH_ON = 0.35
PINCH_OFF = 0.58
FOOT_ZONE = 30
KNEE_ZONE = 26
KICK_COOLDOWN = 0.23

class Game:
    def __init__(self):
        self.best = 0
        self.reset()

    def reset(self):
        self.x, self.y = WIDTH / 2, HEIGHT / 2
        self.vx, self.vy = 0., 0.
        self.r = BALL_RADIUS
        self.held = False
        self.hand_index = None
        self.hand_pos = None
        self.track = deque(maxlen=9)
        self.count = 0
        self.last_kick = -100.
        self.message = 'PINCH BALL TO GRAB / RELEASE TO THROW'
        self.message_until = 0.
        self.ready = True

    def new_round(self, now):
        self.best = max(self.best, self.count)
        self.reset()
        self.message = 'GROUND! TRY AGAIN (R TO RESET)'
        self.message_until = now + 1.4


def landmark_point(lm, idx, w, h):
    p = lm[idx]
    return np.array([float(p.x * w), float(p.y * h)], dtype=np.float32)


def get_hands(result, w, h):
    found = []
    for landmarks in result.hand_landmarks:
        thumb = landmark_point(landmarks, 4, w, h)
        index = landmark_point(landmarks, 8, w, h)
        wrist = landmark_point(landmarks, 0, w, h)
        knuckle = landmark_point(landmarks, 9, w, h)
        center = (thumb + index) * 0.5
        ratio = float(np.linalg.norm(thumb - index) / max(np.linalg.norm(wrist - knuckle), 1.))
        found.append((center, ratio, thumb, index))
    return found


def get_pose_parts(result, w, h):
    parts = []
    if not result.pose_landmarks:
        return parts
    lm = result.pose_landmarks[0]
    for idx, name, kind in ((31, 'L TOE', 'foot'), (32, 'R TOE', 'foot'),
                            (27, 'L ANKLE', 'foot'), (28, 'R ANKLE', 'foot'),
                            (25, 'L KNEE', 'knee'), (26, 'R KNEE', 'knee')):
        p = lm[idx]
        confidence = getattr(p, 'visibility', 1.0) or 0.0
        if confidence >= 0.55 and 0 <= p.x <= 1 and 0 <= p.y <= 1:
            parts.append((name, kind, np.array([p.x*w, p.y*h], dtype=np.float32)))
    return parts


def draw_ball(frame, g):
    c = (int(g.x), int(g.y))
    r = int(g.r)
    cv2.circle(frame, c, r, (245, 245, 245), -1, cv2.LINE_AA)
    cv2.circle(frame, c, r, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.circle(frame, c, max(3, r // 3), (25, 25, 25), -1, cv2.LINE_AA)
    for a in range(5):
        rad = a * (2 * math.pi / 5) - math.pi/2
        p = (int(g.x + r * 0.72 * math.cos(rad)), int(g.y + r * 0.72 * math.sin(rad)))
        cv2.line(frame, c, p, (50, 50, 50), 1, cv2.LINE_AA)


g = Game()
prev_time = time.monotonic()
start_time = prev_time
last_ts = -1
last_parts = {}
window = 'AirBall Football'
cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window, 960, 720)

try:
    with (Vision.HandLandmarker.create_from_options(hand_options) as hand_detector,
          Vision.PoseLandmarker.create_from_options(pose_options) as pose_detector):
        while True:
            ok, frame = cap.read()
            if not ok:
                print('Could not read camera frame.')
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.monotonic()
            dt = min(max(now - prev_time, 1e-3), 0.05)
            prev_time = now
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=np.ascontiguousarray(rgb))
            ts = max(last_ts + 1, int((now - start_time) * 1000))
            last_ts = ts
            hands = get_hands(hand_detector.detect_for_video(image, ts), w, h)
            parts = get_pose_parts(pose_detector.detect_for_video(image, ts), w, h)

            # Grab by pinching close to the ball. Keep identity by nearest previous position.
            if not g.held:
                for i, (center, ratio, thumb, index) in enumerate(hands):
                    if ratio < PINCH_ON and np.linalg.norm(center - (g.x, g.y)) < g.r + 40:
                        g.held = True
                        g.hand_pos = center.copy()
                        g.track.clear()
                        g.track.append((now, float(center[0]), float(center[1])))
                        g.vx = g.vy = 0.
                        break
            else:
                selected = None
                if hands and g.hand_pos is not None:
                    closest = min(hands, key=lambda hand: np.linalg.norm(hand[0] - g.hand_pos))
                    if np.linalg.norm(closest[0] - g.hand_pos) < 150:
                        selected = closest
                if selected is None:
                    # Tracking lost: release without inventing a kick.
                    g.held = False
                    g.vx = g.vy = 0.
                    g.track.clear()
                else:
                    center, ratio, thumb, index = selected
                    g.hand_pos = center.copy()
                    if ratio > PINCH_OFF:
                        if len(g.track) >= 2:
                            t0, x0, y0 = g.track[0]
                            t1, x1, y1 = g.track[-1]
                            duration = max(t1 - t0, 0.025)
                            g.vx = float(np.clip(1.45 * (x1-x0)/duration, -1200, 1200))
                            g.vy = float(np.clip(1.45 * (y1-y0)/duration, -1200, 1200))
                        g.held = False
                        g.track.clear()
                    else:
                        g.x, g.y = float(center[0]), float(center[1])
                        g.track.append((now, g.x, g.y))
                        while len(g.track) > 2 and now - g.track[0][0] > 0.17:
                            g.track.popleft()

            # Physics and foot/knee collision: use landmark upward speed + cooldown.
            if not g.held:
                g.vy += GRAVITY * dt
                g.x += g.vx * dt
                g.y += g.vy * dt
                g.vx *= 0.997 ** (dt * 30)

                candidates = []
                for name, kind, pos in parts:
                    old = last_parts.get(name)
                    if old is None or now - old[0] > 0.20:
                        continue
                    elapsed = max(now - old[0], 1e-3)
                    upward_speed = float((old[1][1] - pos[1]) / elapsed)
                    radius = FOOT_ZONE if kind == 'foot' else KNEE_ZONE
                    dist = float(np.linalg.norm(pos - (g.x, g.y)))
                    if dist < g.r + radius and upward_speed > 50:
                        candidates.append((upward_speed, name, pos))
                if candidates and now - g.last_kick > KICK_COOLDOWN and g.vy > -350:
                    upward_speed, name, pos = max(candidates, key=lambda item: item[0])
                    g.y = min(g.y, float(pos[1] - g.r - 8))
                    g.vy = -min(960., max(420., 320. + 1.3 * upward_speed))
                    g.vx += float(np.clip((g.x - pos[0]) * 1.5, -180, 180))
                    g.count += 1
                    g.last_kick = now
                    g.message = f'{name} JUGGLE! +1'
                    g.message_until = now + 0.5

                if g.x - g.r < 0:
                    g.x, g.vx = g.r, abs(g.vx) * 0.65
                if g.x + g.r > w:
                    g.x, g.vx = w-g.r, -abs(g.vx) * 0.65
                if g.y - g.r < 0:
                    g.y, g.vy = g.r, abs(g.vy) * BOUNCE
                if g.y + g.r >= h:
                    g.new_round(now)
            last_parts = {name: (now, pos.copy()) for name, _, pos in parts}

            for name, kind, pos in parts:
                p = (int(pos[0]), int(pos[1]))
                cv2.circle(frame, p, 7 if kind == 'foot' else 6,
                           (0, 220, 255) if kind == 'foot' else (255, 180, 0), -1)
            for center, ratio, thumb, index in hands:
                cv2.circle(frame, tuple(center.astype(int)), 5, (230, 80, 220), -1)
                cv2.line(frame, tuple(thumb.astype(int)), tuple(index.astype(int)), (255,255,255), 2)

            draw_ball(frame, g)
            cv2.putText(frame, f'JUGGLES: {g.count}   BEST: {g.best}', (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, .75, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame, 'GRABBED' if g.held else 'FREE', (12, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2, cv2.LINE_AA)
            if now < g.message_until:
                cv2.putText(frame, g.message, (12, 92), cv2.FONT_HERSHEY_SIMPLEX,
                            .55, (60,255,100), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Pinch near ball -> throw; kick up with foot/knee | R reset | Q quit',
                        (10, h-14), cv2.FONT_HERSHEY_SIMPLEX, .40, (255,255,255), 1, cv2.LINE_AA)
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                g.reset()
                last_parts.clear()
finally:
    cap.release()
    cv2.destroyAllWindows()
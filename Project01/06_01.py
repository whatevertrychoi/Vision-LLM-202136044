import cv2
import mediapipe as mp
import numpy as np
import time
import math
from collections import deque
from pathlib import Path
# ---- 아래 4개는 이번에 새로 추가한 import (노트북/Jetson 겸용, 녹화, 실행 옵션용) ----
import argparse
import csv
import platform
from datetime import datetime

# AirBall Football: hand throw + foot/knee juggling, 2D webcam prototype.
#
# =====================================================================
# [실행 방법]  반드시 "저장소 루트"에서 실행하는 것을 권장 (모델 경로 때문)
#   Jetson : python 01_project/06_01.py
#            -> 기존과 똑같은 GStreamer USB 카메라 파이프라인(/dev/video0)을 사용
#   노트북 : python 01_project/06_01.py            (Windows면 자동으로 일반 웹캠 사용)
#            python 01_project/06_01.py --camera 1 (웹캠이 여러 개일 때 번호 지정)
#   재생   : python 01_project/06_01.py --video recordings/rec_YYYYMMDD_HHMMSS.mp4
#            -> Jetson에서 녹화한 영상으로 노트북에서 임계값을 튜닝할 때 사용
#   옵션   : --vis 0.30   (발/무릎 visibility 임계값, 기본 0.55)
#            --record     (시작과 동시에 영상+CSV 녹화)
#
# [키]  SPACE: 공 떨어뜨리기 | R: 초기화 | D: 디버그 패널 on/off
#       T: visibility 임계값 순환(0.55->0.40->0.30->0.20) | V: 녹화 on/off | Q: 종료
# =====================================================================

WIDTH, HEIGHT = 640, 480

# ---------------------------------------------------------------------
# [변경] 모델 경로: 기존에는 'src/models/...' 상대경로라서 "어디서 실행했는지"에 따라
# FileNotFoundError가 났다. 이 파일은 <저장소>/01_project/06_01.py 에 있으므로
# parent.parent 가 저장소 루트다. 여러 후보 위치를 순서대로 찾는다.
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL_URLS = {
    'hand_landmarker.task':
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
    'pose_landmarker_full.task':
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task',
}


def find_model(file_name):
    candidates = [
        REPO_ROOT / 'src' / 'models' / 'MediaPipe' / file_name,                    # 저장소 루트 기준(권장)
        Path.cwd() / 'src' / 'models' / 'MediaPipe' / file_name,                   # 기존 방식: 실행 위치 기준
        Path.home() / 'vision-llm' / 'src' / 'models' / 'MediaPipe' / file_name,   # Jetson 기존 작업 폴더
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]  # 못 찾으면 첫 후보를 반환 -> 아래 존재 검사에서 안내 메시지를 낸다


# ---------------------------------------------------------------------
# [추가] 실행 옵션. 옵션 없이 실행하면 기존과 동일하게 동작한다.
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description='AirBall Football (hand throw + foot/knee juggling)')
parser.add_argument('--camera', type=int, default=None,
                    help='웹캠 번호. 지정하면 GStreamer 대신 일반 웹캠(OpenCV 기본 백엔드)으로 연다')
parser.add_argument('--video', type=str, default=None,
                    help='녹화해 둔 영상 파일로 재생 테스트 (같은 이름의 .csv가 있으면 실제 시각 사용)')
parser.add_argument('--vis', type=float, default=0.55,
                    help='발/무릎 visibility 임계값 (기본 0.55 = 기존 값)')
parser.add_argument('--record', action='store_true',
                    help='시작과 동시에 영상+CSV 녹화')
args = parser.parse_args()
video_mode = args.video is not None

HAND_MODEL = find_model('hand_landmarker.task')
POSE_MODEL = find_model('pose_landmarker_full.task')

# 기존 파이프라인은 한 글자도 바꾸지 않았다 (Jetson USB 카메라용, 검증된 설정).
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
        raise FileNotFoundError(
            'Model not found: ' + str(file) + '\n'
            '저장소 루트에서 아래 명령으로 내려받으세요 (Windows는 wget 대신 브라우저/curl.exe):\n'
            '  mkdir -p src/models/MediaPipe\n'
            '  wget ' + MODEL_URLS[file.name] + ' -O src/models/MediaPipe/' + file.name)

# ---------------------------------------------------------------------
# [변경] 카메라 열기. 세 갈래로 나뉜다.
#   1) --video      : 녹화 영상 파일 재생
#   2) Windows 또는 --camera 지정 : 일반 웹캠 (노트북 테스트용)
#   3) 그 외(= Jetson) : 기존 GStreamer 파이프라인 + 기존 오류 메시지 그대로
# ---------------------------------------------------------------------
if video_mode:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError('Video failed to open: ' + args.video)
elif args.camera is not None or platform.system() == 'Windows':
    cam_index = args.camera if args.camera is not None else 0
    # Windows 에서는 DirectShow 가 보통 가장 빨리 열리고 해상도 지정도 잘 먹는다. 실패하면 기본 백엔드로 재시도.
    backends = [getattr(cv2, 'CAP_DSHOW', cv2.CAP_ANY), cv2.CAP_ANY] if platform.system() == 'Windows' \
        else [cv2.CAP_ANY]
    cap = None
    for backend in backends:
        cap = cv2.VideoCapture(cam_index, backend)
        if cap.isOpened():
            break
        cap.release()
    if not cap.isOpened():
        raise RuntimeError('Webcam failed to open. 다른 프로그램이 카메라를 쓰고 있거나 --camera 번호가 다를 수 있습니다.')
    # 공 물리/발 충돌 반경(px)은 640x480 기준으로 맞춘 값이라 같은 해상도로 요청한다.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
else:
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

# ---------------------------------------------------------------------
# [추가] 발/무릎 visibility 임계값.
# 기존에는 get_pose_parts() 안에 0.55 가 하드코딩되어 있었다. 실행 중 T 키로 바꿔가며
# "발이 안 잡히는 게 모델 탓인지, 임계값 탓인지" 바로 비교할 수 있도록 전역 변수로 뺐다.
# ---------------------------------------------------------------------
VIS_MIN = args.vis
VIS_STEPS = (0.55, 0.40, 0.30, 0.20)   # T 키를 누를 때마다 이 순서로 내려가고, 끝에서 처음으로 돌아간다


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
        self.last_kick_name = ''   # [추가] 마지막으로 공을 찬 부위 (디버그 패널 표시용)
        self.message = 'PINCH BALL TO GRAB / RELEASE TO THROW'
        self.message_until = 0.
        # [변경] 기존에는 선언만 되고 쓰이지 않았다. 이제는 "대기 상태"로 쓴다.
        # True 인 동안 공은 중력을 받지 않고 제자리에 떠 있다 -> SPACE 를 누르거나 핀치로 잡았다 놓으면 시작.
        # (기존에는 시작하자마자 공이 떨어져 바닥에 닿고 new_round 가 반복되어, 리프팅 테스트가 불가능했다)
        self.ready = True

    def new_round(self, now):
        self.best = max(self.best, self.count)
        self.reset()   # reset() 이 ready=True 로 만들므로 바닥에 닿으면 다시 대기 상태가 된다
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


# [변경] 기존에는 get_pose_parts() 안에 인라인 튜플로 있던 것을 밖으로 뺐다.
# get_pose_parts()(필터링 후)와 get_pose_debug()(필터링 전)가 같은 목록을 쓰기 위해서다.
# (MediaPipe Pose 랜드마크 번호, 화면 표시 이름, 종류)
POSE_PARTS = ((31, 'L TOE', 'foot'), (32, 'R TOE', 'foot'),
              (27, 'L ANKLE', 'foot'), (28, 'R ANKLE', 'foot'),
              (25, 'L KNEE', 'knee'), (26, 'R KNEE', 'knee'))


def get_pose_parts(result, w, h):
    parts = []
    if not result.pose_landmarks:
        return parts
    lm = result.pose_landmarks[0]
    for idx, name, kind in POSE_PARTS:
        p = lm[idx]
        confidence = getattr(p, 'visibility', 1.0) or 0.0
        # [변경] 0.55 하드코딩 -> VIS_MIN (T 키/--vis 로 조절). 화면 안(0~1) 조건은 기존 그대로.
        if confidence >= VIS_MIN and 0 <= p.x <= 1 and 0 <= p.y <= 1:
            parts.append((name, kind, np.array([p.x*w, p.y*h], dtype=np.float32)))
    return parts


def get_pose_debug(result, w, h):
    """[추가] 필터링 "전"의 원본 값을 돌려준다: {이름: (visibility, presence, x_px, y_px, 화면 안 여부)}.

    get_pose_parts() 는 visibility/화면 밖 조건에서 걸러진 점을 그냥 버린다. 그러면
      (1) 모델이 발을 아예 못 찾은 것인지
      (2) 찾았지만 visibility 가 낮아서(또는 화면 밖이라서) 버려진 것인지
    구분할 수 없다. 이 함수의 결과가 비어 있으면 (1)=사람 자체를 못 찾음, 값은 있는데 get_pose_parts 에서
    빠졌다면 (2) 이다. 화면 밖 점은 모델이 추정해서 좌표를 주기 때문에 x,y 가 0~1 을 벗어날 수 있다.
    """
    info = {}
    if not result.pose_landmarks:
        return info
    lm = result.pose_landmarks[0]
    for idx, name, kind in POSE_PARTS:
        p = lm[idx]
        vis = float(getattr(p, 'visibility', 1.0) or 0.0)
        pres = float(getattr(p, 'presence', 1.0) or 0.0)
        info[name] = (vis, pres, float(p.x * w), float(p.y * h), bool(0 <= p.x <= 1 and 0 <= p.y <= 1))
    return info


def upward_speed_of(name, pos, now):
    """[추가] 이전 프레임 대비 "위로 올라가는 속도"(px/s). 위로 +, 아래로 -.

    v_up = (y_previous - y_current) / dt   (OpenCV 는 아래로 갈수록 y 가 커지므로 부호가 이렇게 된다)
    직전 샘플이 없거나 0.2초보다 오래됐으면 None. (기존 충돌 루프의 계산식을 그대로 함수로 뺀 것)
    """
    old = last_parts.get(name)
    if old is None or now - old[0] > 0.20:
        return None
    elapsed = max(now - old[0], 1e-3)
    return float((old[1][1] - pos[1]) / elapsed)


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


# ---------------------------------------------------------------------
# [추가] 디버그 패널. 문제가 "검출"에서 나는지 "충돌 판정"에서 나는지 눈으로 가르기 위한 것이다.
# ---------------------------------------------------------------------
COL_OK, COL_WARN, COL_BAD = (60, 255, 100), (0, 165, 255), (60, 60, 255)
COL_TXT, COL_HEAD = (255, 255, 255), (0, 255, 255)


def yn(flag):
    return 'Y' if flag else 'N'


def draw_debug(frame, now, g, pose_info, parts, part_dbg, hands, perf):
    h, w = frame.shape[:2]
    fps, hand_ms, pose_ms = perf
    ok_names = [name for name, kind, pos in parts]

    # (1) 화면 위 시각 정보
    #   - 채워진 점은 기존대로 get_pose_parts()를 통과한 점(원본 코드가 이미 그리는 것)
    #   - 주황색 빈 원: 모델은 찾았지만 visibility 가 VIS_MIN 미만이라 버려진 점 (화면 안에 있을 때만)
    #   - 얇은 원: "공 중심이 이 원 안으로 들어와야 충돌 후보"가 되는 반경(g.r + FOOT_ZONE/KNEE_ZONE)
    #     공 중심이 원 안이면 초록, 밖이면 회색
    for name, (vis, pres, x, y, inside) in pose_info.items():
        if name not in ok_names and inside:
            cv2.circle(frame, (int(x), int(y)), 9, COL_WARN, 1, cv2.LINE_AA)
    for name, kind, pos in parts:
        if name in part_dbg:
            _, dist, limit = part_dbg[name]
            ring = COL_OK if dist < limit else (160, 160, 160)
            cv2.circle(frame, (int(pos[0]), int(pos[1])), int(limit), ring, 1, cv2.LINE_AA)

    # (2) 패널에 쓸 줄 만들기. 각 줄 = (칸 목록, 색). 표 형태의 줄은 칸마다 고정 x 위치에 그린다.
    rows = []
    pose_ok = len(pose_info) > 0
    rows.append((['POSE DETECTED: ' + ('YES' if pose_ok else 'NO')], COL_OK if pose_ok else COL_BAD))
    hand_txt = 'HANDS: %d' % len(hands)
    if hands:
        hand_txt += '  (pinch ratio %.2f / ON<%.2f OFF>%.2f)' % (min(hd[1] for hd in hands), PINCH_ON, PINCH_OFF)
    rows.append(([hand_txt], COL_TXT))
    left_foot = 'L TOE' in ok_names or 'L ANKLE' in ok_names
    right_foot = 'R TOE' in ok_names or 'R ANKLE' in ok_names
    rows.append((['LEFT FOOT: ' + ('DETECTED' if left_foot else 'NO') +
                  '   RIGHT FOOT: ' + ('DETECTED' if right_foot else 'NO')],
                 COL_OK if (left_foot or right_foot) else COL_BAD))
    rows.append((['FPS %.1f   HAND %.0fms   POSE %.0fms' % (fps, hand_ms, pose_ms)], COL_TXT))
    rows.append((['VIS_MIN: %.2f   (T: change)' % VIS_MIN], COL_TXT))
    rows.append((['PART', 'VIS', 'STATE', 'VY(up)', 'DIST/LIMIT'], COL_HEAD))
    for idx, name, kind in POSE_PARTS:
        if name not in pose_info:
            rows.append(([name, '--', 'NONE', '--', '--'], COL_BAD))   # 모델이 아예 못 찾음
            continue
        vis, pres, x, y, inside = pose_info[name]
        if not inside:
            state, col = 'OUT', COL_WARN        # 화면 밖으로 추정됨 (카메라에 전신/발이 안 들어옴)
        elif vis < VIS_MIN:
            state, col = 'LOW', COL_WARN        # 화면 안이지만 visibility 부족 -> 임계값에서 걸러짐
        else:
            state, col = 'OK', COL_OK           # 충돌 판정에 실제로 쓰이는 점
        vy_txt, d_txt = '--', '--'
        if state == 'OK' and name in part_dbg:
            vy, dist, limit = part_dbg[name]
            vy_txt = '--' if vy is None else '%+.0f' % vy   # '--' = 직전 프레임에 같은 점이 없어 속도 계산 불가
            d_txt = '%.0f/%.0f' % (dist, limit)
        rows.append(([name, '%.2f' % vis, state, vy_txt, d_txt], col))

    # KICK CONDITION: 실제 발동 조건(원본 코드와 동일)을 가장 가까운 부위 기준으로 하나씩 보여준다.
    #   near : 공 중심과 부위 사이 거리 < g.r + 반경      up : 위로 올라가는 속도 > 50 px/s
    #   cd   : 쿨다운(KICK_COOLDOWN) 지남                 ball : 공이 이미 위로 빠르게 날아가는 중이 아님(g.vy > -350)
    if now - g.last_kick < 0.3:
        rows.append((['KICK CONDITION: TRUE  (FIRED: %s)' % g.last_kick_name], COL_OK))
    elif g.held:
        rows.append((['KICK CONDITION: FALSE  (BALL HELD)'], COL_TXT))
    elif g.ready:
        rows.append((['KICK CONDITION: FALSE  (READY: SPACE=DROP)'], COL_TXT))
    elif not part_dbg:
        rows.append((['KICK CONDITION: FALSE  (NO FOOT/KNEE)'], COL_BAD))
    else:
        nearest = min(part_dbg, key=lambda n: part_dbg[n][1] - part_dbg[n][2])
        vy, dist, limit = part_dbg[nearest]
        f_near = dist < limit
        f_up = vy is not None and vy > 50
        f_cd = now - g.last_kick > KICK_COOLDOWN
        f_ball = g.vy > -350
        all_ok = f_near and f_up and f_cd and f_ball
        rows.append((['NEAREST %s  near:%s up:%s cd:%s ball:%s' %
                      (nearest, yn(f_near), yn(f_up), yn(f_cd), yn(f_ball))], COL_TXT))
        rows.append((['KICK CONDITION: ' + ('TRUE' if all_ok else 'FALSE')], COL_OK if all_ok else COL_TXT))
    rows.append((['JUGGLES: %d   BEST: %d' % (g.count, g.best)], COL_TXT))

    # (3) 반투명 어두운 배경 + 글자 그리기 (전체 프레임 복사 없이 패널 영역만 어둡게 -> Jetson 부담 최소화)
    line_h = 16
    x0, y0 = w - 322, 6
    x1, y1 = w - 4, min(h - 4, y0 + line_h * len(rows) + 8)
    roi = frame[y0:y1, x0:x1]
    roi[:] = (roi * 0.4).astype(np.uint8)
    col_x = [0, 68, 108, 156, 226]   # 표 형태 줄의 칸 시작 x 오프셋
    for i, (cells, color) in enumerate(rows):
        ty = y0 + 16 + i * line_h
        for cx, text in zip(col_x, cells):
            cv2.putText(frame, text, (x0 + 6 + cx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        .38, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------
# [추가] 녹화. Jetson(토요일)에서 한 번 찍어 두면, 노트북에서 --video 로 같은 입력을 몇 번이고 재생하며
# 임계값(VIS_MIN, 50px/s, FOOT_ZONE...)을 튜닝할 수 있다.
#   - 영상(.mp4 또는 .avi): "거울 반전 전" 원본 프레임을 저장 (재생 때 똑같이 반전하므로 입력이 동일)
#   - CSV(.csv, 영상과 같은 이름): 프레임별 실제 시각 t, 공 상태, 6개 부위의 원본 visibility/presence/좌표
#     재생 때는 이 t 를 시계로 쓴다 -> Jetson 이 15fps 로 돌았어도 속도(px/s) 계산이 그대로 재현된다.
# 저장 위치: <저장소>/recordings/   (용량이 크니 .gitignore 에 넣어 두는 걸 권장)
# ---------------------------------------------------------------------
PART_NAMES = [name for idx, name, kind in POSE_PARTS]
CSV_HEADER = (['frame', 't', 'pose', 'fps', 'ball_x', 'ball_y', 'ball_vx', 'ball_vy',
               'held', 'ready', 'count', 'kick', 'vis_min', 'hands', 'pinch_min']
              + [name.replace(' ', '_') + '_' + key for name in PART_NAMES
                 for key in ('vis', 'pres', 'x', 'y')])


class Recorder:
    def __init__(self):
        self.vw = None
        self.f = None
        self.csv = None
        self.path = None
        self.t0 = 0.
        self.n = 0

    @property
    def active(self):
        return self.vw is not None

    def start(self, now):
        out_dir = REPO_ROOT / 'recordings'
        out_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # 코덱 후보를 차례로 시도: Jetson 의 OpenCV 빌드에 따라 mp4v 가 안 열릴 수 있어서 MJPG/XVID 로 대체한다.
        for ext, codec in (('.mp4', 'mp4v'), ('.avi', 'MJPG'), ('.avi', 'XVID')):
            path = out_dir / ('rec_' + stamp + ext)
            vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), 30.0, (WIDTH, HEIGHT))
            if vw.isOpened():
                self.vw, self.path, self.t0, self.n = vw, path, now, 0
                self.f = open(path.with_suffix('.csv'), 'w', newline='')
                self.csv = csv.writer(self.f)
                self.csv.writerow(CSV_HEADER)
                return True
            vw.release()
        return False

    def write(self, raw_frame, row):
        self.vw.write(raw_frame)
        self.csv.writerow(row)
        self.n += 1

    def stop(self):
        if self.vw is None:
            return None
        self.vw.release()
        self.f.close()
        path = self.path
        self.vw = self.f = self.csv = self.path = None
        return path


def load_replay_times(video_path):
    """[추가] 녹화 때 같이 저장된 CSV(영상과 같은 이름)에서 프레임별 시각 t 를 읽는다. 없거나 깨졌으면 []."""
    csv_path = Path(video_path).with_suffix('.csv')
    times = []
    try:
        if csv_path.is_file():
            with open(csv_path, newline='') as f:
                for row in csv.DictReader(f):
                    times.append(float(row['t']))
    except (OSError, ValueError, KeyError):
        return []
    return times


g = Game()
# [변경] 영상 재생 모드에서는 실제 시계 대신 "영상 시계"(프레임 번호/fps 또는 CSV 의 t)를 쓴다. 0초에서 시작.
prev_time = 0.0 if video_mode else time.monotonic()
start_time = prev_time
last_ts = -1
last_parts = {}
window = 'AirBall Football'
cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window, 960, 720)

# ---- [추가] 디버그/녹화/재생용 상태 변수 ----
rec = Recorder()
want_record = args.record          # --record 면 첫 프레임에서 녹화 시작
show_debug = True                  # D 키로 on/off. 발 인식 문제를 잡을 때까지는 기본 on
frame_idx = 0                      # 읽은 프레임 수 (재생 모드에서 CSV 행 번호로 쓰임)
size_warned = False                # 해상도 불일치 경고를 한 번만 출력하기 위한 플래그
video_fps = (cap.get(cv2.CAP_PROP_FPS) or 30.) if video_mode else 30.
replay_times = load_replay_times(args.video) if video_mode else []
perf_prev = time.perf_counter()
fps_est, hand_ms, pose_ms = 0., 0., 0.   # 화면 표시용 FPS / 손 추론 시간 / 포즈 추론 시간 (지수이동평균)

try:
    # [변경] 괄호 with 문법(Python 3.10+) -> 역슬래시 형태. 동작은 같고 노트북의 Python 3.9 이하에서도 돈다.
    with Vision.HandLandmarker.create_from_options(hand_options) as hand_detector, \
            Vision.PoseLandmarker.create_from_options(pose_options) as pose_detector:
        while True:
            loop_t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                print('End of video.' if video_mode else 'Could not read camera frame.')
                break
            # [추가] 해상도가 640x480 이 아니면 맞춘다. 공 위치/중력/발 충돌 반경이 전부 픽셀 단위라
            # 해상도가 다르면 판정이 어긋난다(인수인계 문서 11.7). 16:9 웹캠이면 약간 늘어나므로
            # 가능하면 카메라가 640x480 으로 열리게 두는 게 좋다.
            if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
                if not size_warned:
                    print('Frame size %dx%d -> resized to %dx%d' % (frame.shape[1], frame.shape[0], WIDTH, HEIGHT))
                    size_warned = True
                frame = cv2.resize(frame, (WIDTH, HEIGHT))
            raw_frame = frame                    # [추가] 녹화용: 거울 반전 "전" 원본 (flip 은 새 배열을 만들므로 안전)
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            if video_mode:
                now = replay_times[frame_idx] if frame_idx < len(replay_times) else frame_idx / video_fps
            else:
                now = time.monotonic()
            frame_idx += 1
            dt = min(max(now - prev_time, 1e-3), 0.05)
            prev_time = now
            # [추가] --record 로 시작했다면 첫 프레임에서 녹화를 켠다
            if want_record:
                want_record = False
                if rec.start(now):
                    g.message, g.message_until = 'REC: ' + rec.path.name, now + 2.5
                else:
                    g.message, g.message_until = 'REC FAILED (codec)', now + 2.5
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=np.ascontiguousarray(rgb))
            ts = max(last_ts + 1, int((now - start_time) * 1000))
            last_ts = ts
            # [변경] 손/포즈 추론 시간을 각각 잰다 (Jetson 에서 두 모델을 같이 돌릴 때 FPS 가 어디서 깎이는지 보려고)
            t_a = time.perf_counter()
            hand_result = hand_detector.detect_for_video(image, ts)
            t_b = time.perf_counter()
            pose_result = pose_detector.detect_for_video(image, ts)
            t_c = time.perf_counter()
            hand_ms += 0.1 * ((t_b - t_a) * 1000 - hand_ms)
            pose_ms += 0.1 * ((t_c - t_b) * 1000 - pose_ms)
            hands = get_hands(hand_result, w, h)
            parts = get_pose_parts(pose_result, w, h)          # 필터링 후: 충돌 판정에 실제로 쓰이는 점
            pose_info = get_pose_debug(pose_result, w, h)      # [추가] 필터링 전 원본: 디버그/CSV 용

            # Grab by pinching close to the ball. Keep identity by nearest previous position.
            if not g.held:
                for i, (center, ratio, thumb, index) in enumerate(hands):
                    if ratio < PINCH_ON and np.linalg.norm(center - (g.x, g.y)) < g.r + 40:
                        g.held = True
                        g.ready = False     # [추가] 잡는 순간 대기 상태 해제 (놓으면 바로 던져진다)
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
            # [변경] 조건에 "and not g.ready" 추가: 대기 상태에서는 공이 떨어지지 않는다.
            if not g.held and not g.ready:
                g.vy += GRAVITY * dt
                g.x += g.vx * dt
                g.y += g.vy * dt
                g.vx *= 0.997 ** (dt * 30)

                candidates = []
                for name, kind, pos in parts:
                    # [변경] 기존의 old/elapsed/upward_speed 계산을 upward_speed_of() 로 옮겼다 (계산식 동일).
                    upward_speed = upward_speed_of(name, pos, now)
                    if upward_speed is None:
                        continue
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
                    g.last_kick_name = name   # [추가] 디버그 패널 표시용
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

            # [추가] 디버그용 부위별 측정값: (위로 올라가는 속도, 공까지 거리, 충돌 한계 거리 g.r+반경).
            # 공이 잡혀 있거나 대기 중이어도 계산하므로, 발만 움직여 보면서 속도/거리가 어떻게 나오는지 확인할 수 있다.
            part_dbg = {}
            for name, kind, pos in parts:
                radius = FOOT_ZONE if kind == 'foot' else KNEE_ZONE
                part_dbg[name] = (upward_speed_of(name, pos, now),
                                  float(np.linalg.norm(pos - (g.x, g.y))),
                                  g.r + radius)
            last_parts = {name: (now, pos.copy()) for name, _, pos in parts}

            # [추가] FPS: 실제 처리 속도를 보려고 perf_counter(실제 시계)로 잰다 (영상 재생 모드의 영상 시계와 무관)
            perf_now = time.perf_counter()
            fps_est += 0.1 * (1.0 / max(perf_now - perf_prev, 1e-6) - fps_est)
            perf_prev = perf_now

            # [추가] 녹화 중이면 이번 프레임의 원본 영상과 CSV 한 줄 저장
            if rec.active:
                row = [rec.n, round(now - rec.t0, 4), int(len(pose_info) > 0), round(fps_est, 1),
                       round(g.x, 1), round(g.y, 1), round(g.vx, 1), round(g.vy, 1),
                       int(g.held), int(g.ready), g.count, int(g.last_kick == now), VIS_MIN,
                       len(hands), round(min(hd[1] for hd in hands), 3) if hands else '']
                for name in PART_NAMES:
                    if name in pose_info:
                        vis, pres, px, py, inside = pose_info[name]
                        row += [round(vis, 3), round(pres, 3), round(px, 1), round(py, 1)]
                    else:
                        row += ['', '', '', '']
                rec.write(raw_frame, row)

            for name, kind, pos in parts:
                p = (int(pos[0]), int(pos[1]))
                cv2.circle(frame, p, 7 if kind == 'foot' else 6,
                           (0, 220, 255) if kind == 'foot' else (255, 180, 0), -1)
            for center, ratio, thumb, index in hands:
                cv2.circle(frame, tuple(center.astype(int)), 5, (230, 80, 220), -1)
                cv2.line(frame, tuple(thumb.astype(int)), tuple(index.astype(int)), (255,255,255), 2)

            draw_ball(frame, g)
            if show_debug:
                draw_debug(frame, now, g, pose_info, parts, part_dbg, hands, (fps_est, hand_ms, pose_ms))
            cv2.putText(frame, f'JUGGLES: {g.count}   BEST: {g.best}', (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, .75, (255,255,255), 2, cv2.LINE_AA)
            # [변경] 상태 표시에 READY(대기) 추가
            state_text = 'GRABBED' if g.held else ('READY' if g.ready else 'FREE')
            cv2.putText(frame, state_text, (12, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2, cv2.LINE_AA)
            if now < g.message_until:
                cv2.putText(frame, g.message, (12, 92), cv2.FONT_HERSHEY_SIMPLEX,
                            .55, (60,255,100), 2, cv2.LINE_AA)
            elif g.ready:
                # [추가] 대기 중 안내 (기존 시작 메시지는 message_until=0 이라 한 번도 표시되지 않았다)
                cv2.putText(frame, 'SPACE: DROP BALL  /  PINCH BALL + RELEASE: THROW', (12, 92),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (60,255,100), 1, cv2.LINE_AA)
            if rec.active:
                cv2.circle(frame, (20, 116), 6, (0, 0, 255), -1)
                cv2.putText(frame, 'REC %d' % rec.n, (32, 122), cv2.FONT_HERSHEY_SIMPLEX,
                            .55, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Pinch:grab/throw SPACE:drop R:reset D:debug T:vis V:rec Q:quit',
                        (10, h-14), cv2.FONT_HERSHEY_SIMPLEX, .40, (255,255,255), 1, cv2.LINE_AA)
            cv2.imshow(window, frame)
            # [변경] 영상 재생 모드에서는 원본 fps 에 맞춰 대기시간을 둔다 (카메라 모드는 기존과 같은 1ms)
            wait_ms = 1
            if video_mode:
                wait_ms = max(1, int(1000.0 / video_fps - (time.perf_counter() - loop_t0) * 1000.0))
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                g.reset()
                last_parts.clear()
            # ---- [추가] 새 키 ----
            if key == ord(' ') and g.ready:
                g.ready = False                 # 대기 해제 -> 중력 시작
                g.vx = g.vy = 0.
            if key == ord('d'):
                show_debug = not show_debug
            if key == ord('t'):
                # 현재 값보다 작은 다음 단계로 내린다. 더 작은 게 없으면 처음(0.55)으로 돌아간다.
                lower = [v for v in VIS_STEPS if v < VIS_MIN - 1e-6]
                VIS_MIN = lower[0] if lower else VIS_STEPS[0]
                g.message, g.message_until = 'VIS_MIN = %.2f' % VIS_MIN, now + 1.2
            if key == ord('v'):
                if rec.active:
                    saved = rec.stop()
                    g.message, g.message_until = 'REC SAVED: ' + saved.name, now + 2.5
                elif rec.start(now):
                    g.message, g.message_until = 'REC: ' + rec.path.name, now + 2.5
                else:
                    g.message, g.message_until = 'REC FAILED (codec)', now + 2.5
finally:
    rec.stop()   # [추가] 도중에 종료/예외가 나도 녹화 파일이 깨지지 않게 닫는다
    cap.release()
    cv2.destroyAllWindows()

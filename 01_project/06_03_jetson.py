#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# AirBall Vision Quiz Lab  (06_03 Jetson 버전)
#
# [무엇을 하는 프로그램인가]
#   카메라 영상을 일부러 "망가뜨려서" 보여 주고, 오른쪽 패널의 5지선다 보기(= OpenCV 코드)를 고르면
#   그 코드가 실시간 영상에 바로 적용된다. 정답이면 영상이 복원되고 해설이 뜨고, 오답이면 그 코드의
#   (이상한) 결과가 영상에 그대로 보인다.
#   메뉴:  1 기초 8문제 / 2~6 단계 미션 5개 / 7 전체 / 8 자유 실습
#   단계 미션 = 앞 단계의 "정답 결과"가 다음 단계의 입력이 되는 파이프라인 문제.
#
# [한 프레임이 화면에 나오기까지]
#   카메라 프레임(원본)
#     -> (문제가 있으면) 망가뜨림             ... "문제 화면"
#     -> (보기를 골랐으면) 그 보기를 적용       ... "결과 화면"
#     -> to_view() 로 항상 640x480 BGR uint8 로 통일
#     -> 오른쪽 460px 패널을 옆에 붙여 1100x480 "논리 캔버스" 완성
#     -> App.present() 에서만 확대 배율(scale)을 곱해서 imshow
#   버튼 위치와 마우스 좌표는 전부 "배율을 곱하기 전" 1100x480 좌표계 기준이다.
#
# [06_03.py 에서 Jetson 용으로 바꾼 부분]  퀴즈 내용과 화면은 같고, 아래 4가지만 다르다.
#   1) 카메라   Jetson USB 카메라용 GStreamer 파이프라인(GST_USB)을 05.py / 06.py 와 똑같이 사용
#   2) 경로     손 모델 / 한글 폰트를 "이 파일이 있는 폴더" 기준으로도 찾는다
#   3) 창 크기  xrandr 로 화면 해상도를 읽어 자동 배율을 정한다 (auto_scale)
#   4) 안정성   프레임이 잠깐 안 들어와도 바로 종료하지 않고 3초쯤 기다린다 (READ_FAIL_LIMIT).
#               DISPLAY 가 없거나 카메라가 안 열리면 원인 후보를 알려 준다
#
# [실행]  python3 06_03_jetson.py              (Jetson 데스크톱 터미널, USB 카메라)
#         --demo (가짜 영상) / --video a.mp4 / --image a.png / --csi (CSI 카메라, 실기 검증 전)
#         --set m2 (메뉴 건너뛰기: basic missions all m1~m5) / --mode free / --scale 1.6 / --lang en
#         --hand-model 경로 (hand_landmarker.task 위치를 직접 지정)
#
# [키]    퀴즈  1~5 선택 | SPACE·ENTER 다음 | S 건너뛰기 | M 메뉴 | 결과 화면: R 처음부터, W 틀린 문제만
#         공통  P 퀴즈<->자유 전환 | + - 0 창 크기 | Q·ESC 종료
#         자유  1~9 효과 | M C H K T G V 켜고 끄기 | [ ] 값 조절 | R 초기화
#
# [선택 설치]  mediapipe + hand_landmarker.task 는 미션 5 와 자유 실습의 G / T 키에만 필요하다.
#              없으면 그 항목만 비활성이 되고 나머지는 정상 동작한다.
# ======================================================================

import argparse
import glob
import math
import os
import platform
import random
import re
import sys
import time
from collections import deque
from functools import lru_cache

import cv2
import numpy as np

# Pillow(PIL)는 "한글 글씨"를 그리는 데만 쓴다. cv2.putText 는 ASCII 만 그릴 수 있어서 한글을 넣으면
# '???' 로 깨진다. Pillow 가 없거나 한글 폰트를 못 찾으면 영어 UI 로 자동 전환되고, 프로그램은 그대로
# 동작한다.
try:
    from PIL import Image, ImageDraw, ImageFont
    HAVE_PIL = True
# ImportError 뿐 아니라 어떤 import 오류든 영어 UI 로 안전하게 후퇴한다.
except Exception:
    HAVE_PIL = False


# [화면 레이아웃]  논리 캔버스 1100x480 = 영상(왼쪽 640x480) + 패널(오른쪽 460x480)
# 카메라가 어떤 해상도로 들어와도 fit_frame() 이 640x480 으로 맞춰 주므로, 이 파일의 모든 좌표 계산은
# 640x480 을 기준으로 한다. 창에 보이는 크기를 키우는 배율(scale)은 App.present() 와 App.mouse_cb()
# 두 곳에서만 처리한다.
VID_W, VID_H = 640, 480
PANEL_W = 460
WIN_W, WIN_H = VID_W + PANEL_W, VID_H

# imshow / setMouseCallback 을 같은 이름으로 불러야 같은 창으로 인식된다.
WIN_NAME = "AirBall Vision Quiz Lab (06_03 Jetson)"
FADE_SEC = 0.5

# MediaPipe HandLandmarker 모델 파일의 레포 기준 경로. pip 패키지 안에는 없고 별도의 .task 파일이라 따로
# 받아야 한다. HandTracker._find_model() 이 [--hand-model / 현재 폴더 / 이 파일 폴더 / 한 단계 위(레포
# 루트)] 순서로 찾는다.
HAND_MODEL = "src/models/MediaPipe/hand_landmarker.task"

# Jetson USB 카메라용 GStreamer 파이프라인 (05.py / 06.py 와 같은 문자열). 왼쪽에서 오른쪽으로 읽는다.
#   v4l2src device=/dev/video0   USB 카메라에서 영상을 받음
#   video/x-raw,format=YUY2,...  YUY2 640x480 30fps 로 받겠다고 지정
#   videoconvert -> format=BGR   OpenCV 가 쓰는 BGR 로 변환
#   queue leaky=downstream max-size-buffers=1   가득 차면 오래된 프레임을 버림 (버퍼 1장)
#   appsink drop=true max-buffers=1 sync=false  OpenCV 가 가져가는 출구. 항상 "가장 최신 1장"만 유지
# 마지막 두 줄 덕분에 처리가 느려도 영상이 점점 밀리는(지연이 쌓이는) 일이 없다.
GST_USB = (
    'v4l2src device=/dev/video0 ! '
    'video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! '
    'videoconvert ! video/x-raw,format=BGR ! '
    'queue leaky=downstream max-size-buffers=1 '
    'max-size-bytes=0 max-size-time=0 ! '
    'appsink drop=true max-buffers=1 sync=false'
)

# Jetson CSI(리본 케이블) 카메라용. nvarguscamerasrc 는 NVIDIA 전용 요소라 Jetson 에서만 동작한다.
# memory:NVMM 은 GPU 메모리 버퍼이고, nvvidconv 가 일반 메모리(BGRx)로 옮겨 준다. (실기에서 검증하지
# 못한 파이프라인)
GST_CSI = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# 색은 모두 BGR 순서다. (255, 0, 0) 은 빨강이 아니라 파랑!
C_PANEL = (30, 28, 26)
C_HEADER = (58, 50, 44)
C_BTN = (56, 52, 48)
C_BTN_HOVER = (92, 82, 70)
C_BTN_WRONG = (45, 45, 150)
C_BTN_RIGHT = (60, 140, 60)
C_TEXT = (238, 238, 238)
C_DIM = (160, 160, 160)
C_GOOD = (90, 210, 90)
C_BAD = (80, 80, 245)
C_WARN = (0, 200, 255)
C_ACCENT = (255, 190, 60)

# 샤프닝 커널 3종. cv2.filter2D 는 커널을 영상 위에 올려 "곱하고 더하는" 합성곱 연산을 한다.
#   SHARP_K   노트북의 기본 커널 (중심 5, 상하좌우 -1)       합 = 5 - 4 = 1
#   SHARP_K8  이 퀴즈의 정답 커널 (중심 9, 이웃 8개 -1)      합 = 9 - 8 = 1   더 강하게 선명해짐
#   SHARP_KW  약한 버전 (중심 3, 상하좌우 -0.5)              합 = 3 - 2 = 1
# 커널 원소의 합이 1 이면 "평평한 영역의 밝기"는 그대로 두고, 이웃과 차이가 나는 곳(엣지)만 강조한다.
# (합이 1 보다 크면 전체가 밝아지고, 작으면 어두워진다.)
SHARP_K = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
SHARP_K8 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], np.float32)
SHARP_KW = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]], np.float32)

# 피부색 HSV 범위 (OpenCV 8비트 HSV: H 0~179, S 0~255, V 0~255. H 는 원래 0~360도를 반으로 줄여 저장).
#   H 0~20    빨강~주황~살구색 영역
#   S >= 40   회색·흰색에 가까운(채도가 낮은) 픽셀 제외
#   V >= 70   너무 어두운 픽셀 제외
# 이 범위는 "HSV 로 바꾼 영상"에만 써야 한다. BGR 영상에 그대로 쓰면 엉뚱한 픽셀이 선택된다
# (s_skin_bgr 가 보여 주는 함정). 조명과 피부톤에 따라 달라지므로 정답이 아니라 출발점이다.
SKIN_LO = np.array([0, 40, 70], np.uint8)
SKIN_HI = np.array([20, 255, 255], np.uint8)

# MediaPipe 손 랜드마크 21개: 0 손목 / 1~4 엄지 / 5~8 검지 / 9~12 중지 / 13~16 약지 / 17~20 새끼.
# 각 손가락의 마지막 번호(4, 8, 12, 16, 20)가 손가락 끝(tip). 아래는 뼈대를 선으로 그릴 때 쓰는 연결
# 쌍이다.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9),
    (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

# 손가락이 펴졌는지 재는 관절 3점 (A, B, C). B 를 꼭짓점으로 하는 각도 ABC 를 calculate_angle() 로
# 구한다. 쭉 펴면 세 점이 거의 일직선이라 각도가 180도에 가깝고, 접으면 작아진다. 160도 이상이면
# "펴졌다"고 판정한다.
FINGER_ANGLE_POINTS = ((1, 2, 3), (5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19))
ANGLE_THRESHOLD = 160

# CLAHE 객체는 한 번만 만들어 재사용한다 (프레임마다 새로 만들 필요가 없다).
#   clipLimit=3.0     히스토그램 막대의 높이 한도. 노이즈가 과하게 증폭되는 것을 막는다
#   tileGridSize=8x8  영상을 8x8 칸으로 나눠 칸마다 따로 평활화 (= "국소" 대비 향상)
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def put(img, text, org, scale=0.5, color=C_TEXT, thick=1):
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


# 글씨 테두리: 검정 글씨를 8방향으로 한두 픽셀씩 밀어 먼저 쓰고, 그 위에 본 색으로 한 번 더 쓴다.
# 영상이 밝든 어둡든 글씨가 읽힌다.
def outlined(img, text, org, scale=0.6, color=C_TEXT, thick=2):
    x, y = org
    r = 1 if thick < 2 else 2
    for dx, dy in ((-r, 0), (r, 0), (0, -r), (0, r), (-r, -r), (r, -r), (-r, r), (r, r)):
        cv2.putText(img, text, (x + dx, y + dy), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def text_w(s, scale):
    return cv2.getTextSize(s, FONT, scale, 1)[0][0]


# 버튼 안 코드 문자열을 정해진 폭(max_w)에 맞춰 1~2줄로 나눈다. 돌려주는 값 = (줄들, 글자 크기).
#   1) 큰 글자 크기부터 차례로 시도해서 한 줄에 들어가면 그대로 쓴다
#   2) 안 들어가면 쉼표 / 공백 위치에서 둘로 나눈다. 괄호 깊이(depth)가 얕은 곳을 우선한다
#      (cv2.flip(frame, 0) 처럼 괄호 안 쉼표까지는 허용, 더 깊은 곳은 마지막 수단)
#   3) 그래도 안 되면 가장 작은 글자 크기로 글자 단위로 끊는다
# 매 프레임 버튼 5개에 대해 불리므로 @lru_cache 로 결과를 기억한다. (인자가 전부 해시 가능해야 해서
# scales 는 튜플)
@lru_cache(maxsize=1024)
def fit_lines(text, max_w, scales, indent="", max_lines=2):
    for sc in scales:
        if text_w(text, sc) <= max_w:
            return (text,), sc
    if max_lines >= 2:
        cands, depth = [], 0
        for i, c in enumerate(text):
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
            if c == "," and i < len(text) - 1:
                cands.append((text[:i + 1], text[i + 1:].lstrip(), depth))
            elif c == " " and i > 0 and text[i - 1] != ",":
                cands.append((text[:i].rstrip(), text[i + 1:].lstrip(), depth))
        fallback = None
        for k, sc in enumerate(scales):
            best = None
            for head, tail, d in cands:
                if not head or not tail:
                    continue
                second = indent + tail
                wmax = max(text_w(head, sc), text_w(second, sc))
                if wmax <= max_w and (best is None or (d, wmax) < best[:2]):
                    best = (d, wmax, (head, second))
            if best is None:
                continue
            if best[0] <= 1:
                return best[2], sc
            if fallback is None or best[0] < fallback[0]:
                fallback = (best[0], k, sc, best[2])
        if fallback is not None:
            return fallback[3], fallback[2]
    sc = scales[-1]
    out, cur = [], ""
    for c in text:
        if cur and text_w(cur + c, sc) > max_w:
            out.append(cur)
            cur = indent + c
        else:
            cur += c
    out.append(cur)
    return tuple(out), sc


# OpenCV 에러 메시지는 길고 지저분하다. 핵심 원인 한 줄만 남겨서 영상 위에 보여 주려고 다듬는다.
# (cv2.putText 로 그리기 위해 마지막에 ASCII 로 바꾼다)
def short_err(e, limit=90):
    s = str(e).strip()
    if "error:" in s:
        s = s.split("error:", 1)[1].strip()
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    reasons = [ln.lstrip("> ").strip() for ln in lines[1:] if ln.startswith(">")]
    s = reasons[0] if reasons else " ".join(lines)
    s = s.replace("(-215:Assertion failed) ", "")
    s = re.sub(r" in function '.*'$", "", s)
    s = re.sub(r" \(expected: .*", "", s)
    s = re.sub(r"^Modes other than .*? support only", "findContours supports only", s)
    return ascii_text(s)[:limit]


def clip_text(s, max_w, scale):
    if text_w(s, scale) <= max_w:
        return s
    while len(s) > 1 and text_w(s + "...", scale) > max_w:
        s = s[:-1]
    return s + "..."


def ascii_text(s):
    return str(s).encode("ascii", "replace").decode("ascii")


# 어떤 결과가 나오든(float, 1채널, 크기가 다름, None) 항상 "640x480 BGR uint8" 로 통일한다.
# 캔버스에 그리는 쪽이 이 형식만 가정할 수 있게 해 주는 관문이다.
def to_view(img):
    if img is None:
        return np.zeros((VID_H, VID_W, 3), np.uint8)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[1] != VID_W or img.shape[0] != VID_H:
        img = cv2.resize(img, (VID_W, VID_H))
    return img


# 마스크가 255 인 곳만 원본(src_bgr)을 남긴다. 마스크의 모양/자료형이 제각각이어도 1채널 uint8 로 맞춘
# 뒤 bitwise_and.
def apply_mask(src_bgr, mask):
    m = mask if mask.ndim == 2 else mask[:, :, 0]
    if m.dtype != np.uint8:
        m = (m > 0).astype(np.uint8) * 255
    if m.shape[:2] != src_bgr.shape[:2]:
        m = cv2.resize(m, (src_bgr.shape[1], src_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.bitwise_and(src_bgr, src_bgr, mask=m)


# 카메라 프레임을 640x480 으로 맞춘다. 비율(4:3)이 다르면 늘리지 않고 가운데를 잘라 낸 뒤 resize 한다.
def fit_frame(f):
    h, w = f.shape[:2]
    if (w, h) == (VID_W, VID_H):
        return f
    target = VID_W / VID_H
    if w / h > target:
        nw = int(h * target)
        x0 = (w - nw) // 2
        f = f[:, x0:x0 + nw]
    elif w / h < target:
        nh = int(w / target)
        y0 = (h - nh) // 2
        f = f[y0:y0 + nh]
    return cv2.resize(f, (VID_W, VID_H), interpolation=cv2.INTER_AREA)


# 한글 폰트 파일 찾기. 우선순위: 환경변수 KO_FONT -> 이 파일 옆의 ko_font.* -> OS 기본 폰트 ->
# /usr/share/fonts 검색. 실제로 열어 보고(ImageFont.truetype) 되는 첫 파일을 쓴다. fonts-nanum 을
# 설치하면 NanumGothic / NanumBarunGothic 이 잡힌다. 없으면 None -> TextRenderer 가 영어 UI 로 동작.
def find_korean_font():
    here = os.path.dirname(os.path.abspath(__file__))
    cands = []
    env = os.environ.get("KO_FONT")
    if env:
        cands.append(env)
    for name in ("ko_font.ttf", "ko_font.otf", "ko_font.ttc", "NanumGothic.ttf"):
        cands.append(os.path.join(here, name))
    cands += [
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\NanumGothic.ttf",
        r"C:\Windows\Fonts\gulim.ttc",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/Library/Fonts/NanumGothic.ttf",
    ]
    patterns = [
        "/usr/share/fonts/**/Nanum*Gothic*.ttf",
        "/usr/share/fonts/**/NotoSansCJK*Regular*.ttc",
        "/usr/share/fonts/**/NotoSansCJK*.ttc",
        "/usr/share/fonts/**/NotoSansKR*.*",
        "/usr/share/fonts/**/UnDotum*.ttf",
        "/usr/share/fonts/**/baekmuk*.ttf",
        os.path.expanduser("~/.fonts/**/*Nanum*.ttf"),
    ]
    for p in patterns:
        cands += sorted(glob.glob(p, recursive=True))
    for c in cands:
        if c and os.path.isfile(c):
            try:
                ImageFont.truetype(c, 16)
                return c
            except Exception:
                continue
    return None


# 글씨를 그리는 방식:  글자를 "커버리지 맵"(0~1 농도의 흑백 이미지)으로 먼저 만들고, paste_cov() 가 그
# 농도만큼 색을 섞어 붙인다. 한글 모드는 Pillow 로, 영어 모드는 cv2.putText 로 맵을 만들 뿐 붙이는 쪽은
# 같다.
class TextRenderer:
    def __init__(self, lang="auto"):
        self.font_path = None
        if HAVE_PIL and lang != "en":
            self.font_path = find_korean_font()
        self.ko = self.font_path is not None
        self._fonts = {}

    def font(self, size):
        if size not in self._fonts:
            self._fonts[size] = ImageFont.truetype(self.font_path, size)
        return self._fonts[size]

    def _width(self, s, size):
        if self.ko:
            f = self.font(size)
            try:
                return f.getlength(s)
            except AttributeError:
                return f.getsize(s)[0]
        return cv2.getTextSize(s, FONT, size / 36.0, 1)[0][0]

    def wrap(self, text, size, max_w):
        lines = []
        for para in text.split("\n"):
            cur = ""
            for word in para.split(" "):
                trial = word if not cur else cur + " " + word
                if self._width(trial, size) <= max_w:
                    cur = trial
                    continue
                if cur:
                    lines.append(cur)
                    cur = ""
                if self._width(word, size) <= max_w:
                    cur = word
                else:
                    buf = ""
                    for ch in word:
                        if self._width(buf + ch, size) <= max_w:
                            buf += ch
                        else:
                            lines.append(buf)
                            buf = ch
                    cur = buf
            lines.append(cur)
        return lines

    # text 를 폭 w 에 맞춰 줄바꿈(wrap)해서 (커버리지 맵, 높이) 를 돌려준다. 줄 높이 = 글자 크기 x 1.4.
    def block(self, text, w, size, max_lines=None):
        lines = self.wrap(text, size, w)
        if max_lines:
            lines = lines[:max_lines]
        lh = int(size * 1.4)
        h = max(lh * len(lines), 1) + 2
        if self.ko:
            img = Image.new("L", (w, h), 0)
            d = ImageDraw.Draw(img)
            f = self.font(size)
            for i, ln in enumerate(lines):
                d.text((0, i * lh), ln, font=f, fill=255)
            cov = np.asarray(img, np.float32) / 255.0
        else:
            cov8 = np.zeros((h, w), np.uint8)
            for i, ln in enumerate(lines):
                cv2.putText(cov8, ln, (0, i * lh + size), FONT, size / 36.0, 255, 1, cv2.LINE_AA)
            cov = cov8.astype(np.float32) / 255.0
        return cov, h

    def draw(self, img, text, x, y, w, size=16, color=C_TEXT, max_lines=None):
        cov, h = self.block(text, w, size, max_lines)
        paste_cov(img, cov, x, y, color)
        return y + h


# 알파 합성:  결과 = 원래 픽셀 x (1 - 농도) + 글자색 x 농도.  화면 가장자리를 넘어가는 부분은 잘라서
# 처리한다.
def paste_cov(img, cov, x, y, color):
    h, w = cov.shape[:2]
    y2, x2 = min(y + h, img.shape[0]), min(x + w, img.shape[1])
    if y2 <= y or x2 <= x:
        return
    c = cov[:y2 - y, :x2 - x, None]
    sub = img[y:y2, x:x2].astype(np.float32)
    col = np.array(color, np.float32)
    img[y:y2, x:x2] = (sub * (1.0 - c) + col * c).astype(np.uint8)


# --demo 용 가짜 영상. 카메라 없이도 화면 배치와 클릭을 점검할 수 있다. 배경 그라데이션 위에서 얼굴 모양
# 타원과 초록 원이 t(초)에 따라 움직이고, 파란 사각형은 고정이다.
def synthetic_frame(t):
    f = np.zeros((VID_H, VID_W, 3), np.uint8)
    xs = np.linspace(0, 1, VID_W, dtype=np.float32)[None, :]
    ys = np.linspace(0, 1, VID_H, dtype=np.float32)[:, None]
    f[:, :, 0] = (60 + 90 * xs).astype(np.uint8)
    f[:, :, 1] = (70 + 80 * ys).astype(np.uint8)
    f[:, :, 2] = (60 + 40 * (1 - xs)).astype(np.uint8)
    cx = int(VID_W * (0.5 + 0.28 * math.sin(t * 1.3)))
    cy = int(VID_H * (0.5 + 0.22 * math.sin(t * 0.9 + 1.0)))
    cv2.ellipse(f, (cx, cy), (70, 90), 0, 0, 360, (140, 170, 230), -1, cv2.LINE_AA)
    cv2.circle(f, (cx - 25, cy - 20), 8, (40, 40, 40), -1, cv2.LINE_AA)
    cv2.circle(f, (cx + 25, cy - 20), 8, (40, 40, 40), -1, cv2.LINE_AA)
    gx = int(VID_W * (0.2 + 0.1 * math.cos(t * 0.7)))
    cv2.circle(f, (gx, 380), 40, (60, 200, 60), -1, cv2.LINE_AA)
    cv2.rectangle(f, (470, 330), (600, 440), (40, 90, 200), 3)
    cv2.putText(f, "DEMO", (20, 460), FONT, 1.0, (230, 230, 230), 2, cv2.LINE_AA)
    return f


# 카메라가 안 열렸을 때 원인 후보를 알려 준다: OpenCV 가 GStreamer 없이 빌드됐는지(getBuildInformation),
# /dev/video0 가 있는지, 다른 프로그램이 카메라를 쓰고 있는지.
def camera_help(csi=False):
    lines = ["카메라를 열지 못했습니다 (GStreamer 파이프라인: %s)." % ("CSI nvarguscamerasrc" if csi else "USB /dev/video0")]
    try:
        m = re.search(r"GStreamer:\s*(\S+)", cv2.getBuildInformation())
        gst = m.group(1) if m else "?"
    except Exception:
        gst = "?"
    if gst.upper().startswith("NO"):
        lines.append("  - 이 OpenCV(%s)는 GStreamer 없이 빌드됐습니다. 05.py / 06.py 가 돌아가는 같은 환경(JetPack OpenCV)에서 실행하세요." % cv2.__version__)
    if not csi and platform.system() == "Linux" and not os.path.exists("/dev/video0"):
        lines.append("  - /dev/video0 가 없습니다. USB 카메라 연결을 확인하세요:  ls /dev/video*")
    lines.append("  - 다른 프로그램(05.py / 06.py 등)이 카메라를 쓰고 있지 않은지 확인하세요.")
    lines.append("  - 장치 확인: v4l2-ctl --list-devices      GStreamer 만 시험: gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! autovideosink")
    lines.append("  - 카메라 없이 화면만 확인하려면:  python 06_03_jetson.py --demo")
    return "\n".join(lines)


# 입력을 하나로 감싼 클래스. kind = "camera" | "demo" | "image" | "video".
# read() 는 항상 (성공 여부, 640x480 프레임) 을 돌려주므로 main 루프는 입력이 무엇인지 몰라도 된다.
class FrameSource:
    def __init__(self, args):
        self.kind = "camera"
        self.cap = None
        self.image = None
        self.t0 = time.monotonic()
        if args.demo:
            self.kind = "demo"
        elif args.image:
            self.image = cv2.imread(args.image)
            if self.image is None:
                raise RuntimeError("Cannot read image: %s" % args.image)
            self.kind = "image"
        elif args.video:
            self.cap = cv2.VideoCapture(args.video)
            if not self.cap.isOpened():
                raise RuntimeError("Cannot open video: %s" % args.video)
            self.kind = "video"
        else:
            self.cap = self._open_camera(args)

    # GStreamer 파이프라인으로만 연다 (cv2.CAP_GSTREAMER). 못 열면 camera_help() 내용을 담아 예외를
    # 던지고, main() 이 그것을 출력한 뒤 종료한다.
    @staticmethod
    def _open_camera(args):
        pipe = GST_CSI if args.csi else GST_USB
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(camera_help(args.csi))
        print("[camera] GStreamer pipeline opened (%s)" % ("CSI nvarguscamerasrc" if args.csi else "USB /dev/video0"))
        return cap

    # 영상 파일은 끝나면 처음(0번 프레임)으로 되감아 계속 반복한다. 카메라가 프레임을 못 주면 (False,
    # None).
    def read(self):
        if self.kind == "demo":
            return True, synthetic_frame(time.monotonic() - self.t0)
        if self.kind == "image":
            return True, fit_frame(self.image.copy())
        ok, f = self.cap.read()
        if not ok and self.kind == "video":
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = self.cap.read()
        if not ok:
            return False, None
        return True, fit_frame(f)

    def release(self):
        if self.cap is not None:
            self.cap.release()


# 노이즈 / 흰 점 패턴은 매 프레임 새로 만들면 비싸므로, 6장씩 미리 만들어 두고 프레임마다 하나를
# 무작위로 골라 쓴다. (seed 7 고정이라 실행할 때마다 같은 패턴 6장이 나온다)
_rng = np.random.default_rng(7)
_NOISE = []
_SPECK = []


# 정규분포 노이즈 (평균 0, 표준편차 42). int16 으로 만드는 이유: uint8 영상에 음수 값을 더해야 하고,
# uint8 끼리 더하면 255 를 넘을 때 값이 한 바퀴 돌아 버리기(오버플로) 때문.
def _noise_pool():
    if not _NOISE:
        for _ in range(6):
            _NOISE.append(_rng.normal(0, 42, (VID_H, VID_W, 3)).astype(np.int16))
    return _NOISE


# 마스크에 섞을 "작은 흰 점" 패턴: 0.6% 확률로 점을 찍고 2x2 로 키운다. 모폴로지 OPEN 문제의 재료다.
def _speck_pool():
    if not _SPECK:
        k = np.ones((2, 2), np.uint8)
        for _ in range(6):
            seeds = (_rng.random((VID_H, VID_W)) < 0.006).astype(np.uint8) * 255
            _SPECK.append(cv2.dilate(seeds, k))
    return _SPECK


# BGR -> HSV 로 바꾼 뒤 피부색 범위(SKIN_LO ~ SKIN_HI)에 드는 픽셀만 255 로 남긴 마스크.
def skin_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, SKIN_LO, SKIN_HI)


# 노이즈 크기 추정용 커널 (Immerkaer 방식). 평평한 곳과 일정한 경사(평면)에서는 합성곱 결과가 0 이
# 되도록 짜여 있어서, 결과에 남는 것은 거의 노이즈뿐이다. metric_noise() 가 그 절댓값 평균으로
# 표준편차를 추정한다.
_NOISE_K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def _gray_of(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# 선명도 = 라플라시안(2차 미분)의 분산. 엣지가 뚜렷할수록 값이 크다. 화면 왼쪽 위의 "Sharpness" 숫자.
def metric_sharp(img):
    return float(cv2.Laplacian(_gray_of(img), cv2.CV_32F).var())


# 합성곱 결과 절댓값의 평균 x sqrt(pi/2) / 6  =  노이즈 표준편차의 추정치. 테두리 1픽셀은
# 제외한다([1:-1]).
def metric_noise(img):
    g = _gray_of(img).astype(np.float32)
    c = cv2.filter2D(g, -1, _NOISE_K, borderType=cv2.BORDER_REFLECT)[1:-1, 1:-1]
    return float(math.sqrt(math.pi / 2.0) / 6.0 * np.abs(c).mean())


def metric_contrast(img):
    return float(_gray_of(img).std())


# 화면에 숫자로 보여 줄 측정값. 각 항목 = (이름, 함수, 값이 클수록 좋은가).
# 정답을 골랐을 때 "좋아졌는지"를 초록색 / 주황색으로 구분하는 데 쓴다 (Quiz._metric).
METRICS = {
    "sharp": ("Sharpness", metric_sharp, True),
    "noise": ("Noise level", metric_noise, False),
    "contrast": ("Contrast", metric_contrast, True),
}


# p_*  = problem. 영상을 일부러 망가뜨리는 함수 (문제).
# s_*  = solution candidate. 보기로 나오는 해결 후보 함수. 정답도 오답도 전부 s_* 이다.
def p_flip_ud(f):
    return cv2.flip(f, 0)


def p_swap_rb(f):
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)


def p_blur(f):
    return cv2.GaussianBlur(f, (9, 9), 0)


def p_noise(f):
    pool = _noise_pool()
    n = pool[random.randrange(len(pool))]
    return np.clip(f.astype(np.int16) + n, 0, 255).astype(np.uint8)


def p_dark(f):
    return cv2.convertScaleAbs(f, alpha=0.35, beta=10)


def p_noisy_mask(f):
    pool = _speck_pool()
    return cv2.bitwise_or(skin_mask(f), pool[random.randrange(len(pool))])


# 짧은 보기 함수들은 한 줄씩 모아 둔다. 보기 하나 = (화면에 보이는 코드 문자열, 실제로 실행되는 s_*
# 함수) 쌍이다. 오답 보기도 진짜로 실행된다. 그래서 틀린 결과가 영상에 그대로 보인다.
def s_flip0(f): return cv2.flip(f, 0)
def s_flip1(f): return cv2.flip(f, 1)
def s_flipm1(f): return cv2.flip(f, -1)
def s_slice_lr(f): return f[:, ::-1].copy()
def s_rot90(f): return cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE)
def s_bgr2rgb(f): return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
def s_bgr2gray(f): return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
def s_bgr2hsv(f): return cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
def s_bgr2lab(f): return cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
def s_gauss11(f): return cv2.GaussianBlur(f, (11, 11), 0)
def s_gauss9(f): return cv2.GaussianBlur(f, (9, 9), 0)
def s_gauss21(f): return cv2.GaussianBlur(f, (21, 21), 0)
def s_box21(f): return cv2.blur(f, (21, 21))
def s_gauss1(f): return cv2.GaussianBlur(f, (1, 1), 0)
def s_sharp(f): return cv2.filter2D(f, -1, SHARP_K8)
def s_canny_bgr(f): return cv2.Canny(f, 100, 200)


def s_canny_gray(f):
    return cv2.Canny(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), 100, 200)


# 함정: cv2.Canny 는 uint8 만 받는다. float32 로 바꿔 넣으면 에러가 나고, 그 에러가 화면에 표시된다.
def s_canny_f32(f):
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray.astype(np.float32) / 255., 100, 200)


def s_blur_gray(f):
    return cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (11, 11), 0)


def s_sharp_gray(f):
    return cv2.filter2D(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), -1, SHARP_K8)


def s_skin_hsv(f):
    return cv2.inRange(cv2.cvtColor(f, cv2.COLOR_BGR2HSV), SKIN_LO, SKIN_HI)


# 함정: HSV 범위를 BGR 영상에 그대로 적용. 에러는 안 나지만 엉뚱한 픽셀이 선택된다.
def s_skin_bgr(f):
    return cv2.inRange(f, SKIN_LO, SKIN_HI)


def s_blue_hsv(f):
    return cv2.inRange(cv2.cvtColor(f, cv2.COLOR_BGR2HSV),
                       np.array([100, 40, 70], np.uint8), np.array([130, 255, 255], np.uint8))


def s_thresh_gray(f):
    return cv2.threshold(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1]


# 모폴로지용 5x5 타원 커널. 한 번만 만든다.
_K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def s_open(m): return cv2.morphologyEx(m, cv2.MORPH_OPEN, _K5)
def s_close(m): return cv2.morphologyEx(m, cv2.MORPH_CLOSE, _K5)
def s_dilate(m): return cv2.dilate(m, _K5)
def s_gauss_mask(m): return cv2.GaussianBlur(m, (5, 5), 0)
def s_canny_mask(m): return cv2.Canny(m, 100, 200)


# 정답 방식: BGR -> LAB 로 바꿔서 밝기 채널 L 에만 CLAHE 를 적용하고 다시 BGR 로 합친다.
# 색 정보(a, b)는 건드리지 않으므로 색이 틀어지지 않는다.
def clahe_on_L(f):
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _CLAHE.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# 오답 방식: B, G, R 채널을 각각 평활화하면 채널마다 대비가 달라져서 색이 틀어진다.
def s_eq_bgr(f):
    b, g, r = cv2.split(f)
    return cv2.merge([cv2.equalizeHist(b), cv2.equalizeHist(g), cv2.equalizeHist(r)])


# 기초 퀴즈 8문제의 데이터. 문제 하나 = dict:
#   unit, unit_name   단원 표시                 ko, en   문제 문장 (한글 / 영어 UI)
#   setup             문제 아래에 보여 줄 "이미 있는 변수" 코드 줄들
#   problem           영상을 망가뜨리는 p_* 함수 (None 이면 원본 그대로)
#   metric            화면에 보여 줄 측정값 ("sharp" / "noise" / "contrast", 없으면 생략)
#   choices           보기 5개. 각 보기 = 6개짜리 튜플 (code, fn, ok, why_ko, why_en, kind)
#       code   버튼에 적힌 코드 문자열         fn      실제로 실행할 s_* 함수
#       ok     정답이면 True                   why_*   해설 (한글 / 영어)
#       kind   결과를 화면에 올리는 방법
#                "image"  fn(망가진 프레임)의 결과를 그대로 영상으로
#                "mask"   fn 이 만든 마스크를 망가진 영상에 씌워서 (apply_mask)
#                "st"     단계 미션용. fn(상태, ctx) 를 부름 (아래 미션 쪽 설명 참고)
def build_questions():
    sharp_setup = "# sharp_kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]], np.float32)"
    qs = []

    qs.append(dict(
        unit="A", unit_name="Image adjustment",
        ko="화면이 위아래로 뒤집혀 보입니다. 원래대로 돌리는 코드는?",
        en="The view is upside down. Which line restores it?",
        setup=[], problem=p_flip_ud,
        choices=[
            ("cv2.flip(frame, 0)", s_flip0, True,
             "flipCode 0은 상하 반전입니다. 같은 반전을 한 번 더 하면 원래대로 돌아옵니다. (1=좌우, 0=상하, -1=둘 다)",
             "flipCode 0 = vertical flip. Flipping again restores it. (1=horizontal, 0=vertical, -1=both)", "image"),
            ("cv2.flip(frame, 1)", s_flip1, False,
             "flipCode 1은 좌우 반전입니다. 위아래는 그대로 뒤집혀 있어요.",
             "flipCode 1 is a horizontal flip; it stays upside down.", "image"),
            ("cv2.flip(frame, -1)", s_flipm1, False,
             "-1은 상하와 좌우를 모두 뒤집습니다. 결과적으로 좌우만 뒤집힌 상태가 돼요.",
             "-1 flips both axes, leaving a left-right mirrored view.", "image"),
            ("frame[:, ::-1]", s_slice_lr, False,
             "열 순서를 뒤집는 NumPy 좌우 반전입니다. 위아래는 그대로예요.",
             "Reverses the columns (left-right). Still upside down.", "image"),
            ("cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)", s_rot90, False,
             "90도 회전이라 가로세로가 바뀝니다. 뒤집힘을 바로잡는 코드가 아니에요.",
             "A 90-degree rotation swaps width/height; it does not fix a flip.", "image"),
        ]))

    qs.append(dict(
        unit="B", unit_name="Color space",
        ko="피부가 파랗게 보입니다. 채널 순서(B, R)가 뒤바뀐 상태예요. 바로잡는 코드는?",
        en="Skin looks blue: the B and R channels are swapped. Which line fixes it?",
        setup=[], problem=p_swap_rb,
        choices=[
            ("cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)", s_bgr2rgb, True,
             "B와 R 채널 순서를 맞바꿉니다. 한 번 더 바꾸면 원래 색이에요. OpenCV는 BGR, matplotlib/skimage는 RGB입니다.",
             "Swaps the B and R channels; swapping again restores the colors. OpenCV is BGR, matplotlib/skimage is RGB.", "image"),
            ("cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)", s_bgr2gray, False,
             "흑백 변환입니다. 색 정보가 사라질 뿐 파란 색조 문제를 고친 게 아니에요.",
             "Grayscale just drops color; it does not fix the swapped channels.", "image"),
            ("cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)", s_bgr2hsv, False,
             "HSV 값을 BGR인 것처럼 그리면 색이 엉뚱해집니다.",
             "Drawing HSV values as if they were BGR gives wrong colors.", "image"),
            ("cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)", s_bgr2lab, False,
             "LAB 값을 BGR인 것처럼 그리면 색이 엉뚱해집니다.",
             "Drawing LAB values as if they were BGR gives wrong colors.", "image"),
            ("cv2.flip(frame, 1)", s_flip1, False,
             "좌우 반전은 색과 무관합니다.",
             "A horizontal flip has nothing to do with color.", "image"),
        ]))

    qs.append(dict(
        unit="C", unit_name="Filtering",
        ko="영상이 많이 흐릿합니다. 선명하게 만드는 코드는? (화면 왼쪽 위의 선명도 숫자도 같이 보세요)",
        en="The view is very blurry. Which line sharpens it? (watch the Sharpness number, top-left)",
        setup=[sharp_setup], problem=p_blur, metric="sharp",
        choices=[
            ("cv2.filter2D(frame, -1, sharp_kernel)", s_sharp, True,
             "중심은 키우고(9) 이웃 8개는 빼는(-1) 커널이라 픽셀과 이웃의 차이(엣지)가 크게 강조됩니다. 커널 합이 1이라 밝기는 유지돼요. (노트북의 5짜리 커널보다 강한 버전)",
             "Center 9, eight neighbors -1: it strongly boosts the difference between a pixel and its neighbors (edges). Kernel sum is 1, so brightness is kept. (A stronger cousin of the notebook's 5-kernel.)", "image"),
            ("cv2.GaussianBlur(frame, (21, 21), 0)", s_gauss21, False,
             "주변 평균을 내는 블러라 오히려 훨씬 더 흐려집니다. 선명도 숫자가 떨어지는 걸 보세요.",
             "A blur averages neighbors, so it gets much blurrier. Watch the sharpness number drop.", "image"),
            ("cv2.blur(frame, (21, 21))", s_box21, False,
             "Box 블러도 평균이라 더 흐려집니다. (거의 뭉개진 덩어리)",
             "A box blur is also an average; blurrier again.", "image"),
            ("cv2.Canny(frame, 100, 200)", s_canny_bgr, False,
             "엣지만 뽑는 함수라 선명해지는 게 아니라 윤곽선만 남습니다.",
             "Canny keeps only edge lines; it does not sharpen the image.", "image"),
            ("cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)", s_bgr2gray, False,
             "흑백으로 바뀔 뿐 선명도는 그대로입니다.",
             "It only turns the image gray; sharpness is unchanged.", "image"),
        ]))

    qs.append(dict(
        unit="C", unit_name="Filtering",
        ko="화면이 자글자글한 노이즈로 가득합니다. 노이즈를 줄이는 코드는? (왼쪽 위 노이즈 수치도 같이 보세요)",
        en="The view is full of grainy noise. Which line reduces it? (watch the Noise number, top-left)",
        setup=[sharp_setup], problem=p_noise, metric="noise",
        choices=[
            ("cv2.GaussianBlur(frame, (9, 9), 0)", s_gauss9, True,
             "주변을 가우시안 가중 평균하면 랜덤 노이즈가 크게 줄어듭니다. 엣지 검출 전에 블러부터 하는 이유이기도 해요. 커널이 클수록 더 많이 지워지지만 그만큼 흐려집니다.",
             "A Gaussian weighted average removes most random noise. That is why we blur before edge detection. A bigger kernel removes more but blurs more.", "image"),
            ("cv2.filter2D(frame, -1, sharp_kernel)", s_sharp, False,
             "샤프닝은 픽셀 차이를 키우는 필터라 노이즈도 같이 증폭됩니다.",
             "Sharpening boosts pixel differences, so it amplifies the noise too.", "image"),
            ("cv2.Canny(frame, 100, 200)", s_canny_bgr, False,
             "노이즈 점마다 엣지로 잡혀서 화면이 더 지저분해집니다.",
             "Every noisy pixel becomes an edge; it looks even messier.", "image"),
            ("cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)", s_bgr2lab, False,
             "색공간을 바꿔 그리면 색만 이상해지고 노이즈는 그대로예요.",
             "Changing color space only distorts colors; the noise stays.", "image"),
            ("cv2.flip(frame, 1)", s_flip1, False,
             "좌우 반전은 노이즈와 무관합니다.",
             "A horizontal flip does nothing about noise.", "image"),
        ]))

    qs.append(dict(
        unit="D", unit_name="Edges",
        ko="경계선(엣지)만 뽑아내고 싶습니다. 올바른 코드는?",
        en="Extract only the edges. Which code is right?",
        setup=["# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)", sharp_setup], problem=None,
        choices=[
            ("cv2.Canny(gray, 100, 200)", s_canny_gray, True,
             "Canny는 uint8 단일 채널(흑백) 영상용 엣지 검출기입니다. 낮은/높은 임계값 100, 200으로 엣지를 정해요.",
             "Canny is an edge detector for uint8 grayscale images; 100/200 are the low/high thresholds.", "image"),
            ("cv2.Canny(gray.astype(np.float32) / 255., 100, 200)", s_canny_f32, False,
             "Canny는 uint8 전용입니다. float32로 바꾸면 에러가 나요(노트북에 나왔던 함정).",
             "Canny only takes uint8. float32 input raises an error (the trap from the notebook).", "image"),
            ("cv2.GaussianBlur(gray, (11, 11), 0)", s_blur_gray, False,
             "블러는 경계를 뭉개서 오히려 엣지가 사라집니다.",
             "Blur smears boundaries; edges fade away.", "image"),
            ("cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)", s_bgr2hsv, False,
             "색공간 변환이지 엣지 검출이 아닙니다.",
             "That is a color-space conversion, not edge detection.", "image"),
            ("cv2.filter2D(gray, -1, sharp_kernel)", s_sharp_gray, False,
             "경계를 강조해 선명하게 할 뿐, 엣지만 따로 뽑아 주지는 않아요.",
             "It enhances edges but still returns the whole image, not just edges.", "image"),
        ]))

    qs.append(dict(
        unit="E", unit_name="Color segmentation",
        ko="화면에서 내 피부색(얼굴·손)만 남기고 싶습니다. 올바른 코드는? (카메라에 얼굴이나 손을 보여 주세요)",
        en="Keep only skin-colored pixels (face, hands). Which code is right? (show your face or hand)",
        setup=["# hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)",
               "# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)",
               "# mask is applied: cv2.bitwise_and(frame, frame, mask=mask)"], problem=None,
        choices=[
            ("cv2.inRange(hsv, (0, 40, 70), (20, 255, 255))", s_skin_hsv, True,
             "색 판단은 색상과 밝기가 분리된 HSV에서 합니다. 피부는 H가 0~20 부근(주황~살구)이라 그 범위만 남아요. (나무·베이지색 배경도 같이 남을 수 있어요.)",
             "Judge color in HSV, where hue and brightness are separate. Skin hue is about 0-20 (orange). Wood or beige walls can match too.", "mask"),
            ("cv2.inRange(frame, (0, 40, 70), (20, 255, 255))", s_skin_bgr, False,
             "BGR 값에 HSV 범위를 그대로 쓰면 엉뚱한 픽셀이 선택됩니다. 먼저 HSV로 변환해야 해요.",
             "Using HSV ranges on BGR values picks the wrong pixels. Convert to HSV first.", "mask"),
            ("cv2.inRange(hsv, (100, 40, 70), (130, 255, 255))", s_blue_hsv, False,
             "H 100~130은 파랑 계열입니다. 피부색 범위가 아니에요.",
             "H 100-130 is blue, not skin.", "mask"),
            ("cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]", s_thresh_gray, False,
             "밝기 기준 이진화라 색과 무관하게 밝은 영역이 남습니다.",
             "It thresholds brightness, regardless of color.", "mask"),
            ("cv2.Canny(frame, 100, 200)", s_canny_bgr, False,
             "윤곽선만 남기는 함수예요. 색으로 고르지 않습니다.",
             "Canny keeps outlines only; it does not select by color.", "image"),
        ]))

    qs.append(dict(
        unit="F", unit_name="Morphology",
        ko="피부색 마스크에 작은 흰 점 노이즈가 가득합니다. 점은 지우고 큰 덩어리는 살리는 코드는?",
        en="The skin mask is full of tiny white specks. Which line removes them but keeps the big blob?",
        setup=["# mask = skin mask + white speck noise",
               "# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))"], problem=p_noisy_mask,
        choices=[
            ("cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)", s_open, True,
             "Open은 침식 후 팽창입니다. 작은 흰 점은 침식에서 사라지고, 큰 덩어리는 팽창으로 원래 크기를 회복해요.",
             "Open = erode then dilate. Tiny white specks vanish in the erosion; big blobs grow back.", "image"),
            ("cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)", s_close, False,
             "Close(팽창 후 침식)는 작은 검은 구멍을 메웁니다. 흰 점 노이즈는 그대로 남아요.",
             "Close (dilate then erode) fills small black holes; white specks stay.", "image"),
            ("cv2.dilate(mask, kernel)", s_dilate, False,
             "팽창은 흰 영역을 키우므로 점 노이즈가 오히려 커집니다.",
             "Dilation grows white areas, so the specks get bigger.", "image"),
            ("cv2.GaussianBlur(mask, (5, 5), 0)", s_gauss_mask, False,
             "블러는 점을 회색 얼룩으로 번지게 할 뿐 지우지 못합니다.",
             "Blur smears specks into gray blotches instead of removing them.", "image"),
            ("cv2.Canny(mask, 100, 200)", s_canny_mask, False,
             "마스크의 테두리만 뽑는 함수라 점 노이즈도 테두리로 남습니다.",
             "It extracts mask outlines, so every speck remains as an outline.", "image"),
        ]))

    qs.append(dict(
        unit="G", unit_name="Enhancement",
        ko="화면이 어둡고 뿌옇습니다. 색은 유지하면서 대비를 살리는 방법은?",
        en="The view is dark and flat. Which approach restores contrast while keeping colors?",
        setup=["# clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))",
               "# L, a, b = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))"],
        problem=p_dark, metric="contrast",
        choices=[
            ("cv2.cvtColor(cv2.merge([clahe.apply(L), a, b]), cv2.COLOR_LAB2BGR)", clahe_on_L, True,
             "LAB의 밝기(L) 채널에만 CLAHE를 적용하면 색은 유지한 채 구간별로 대비가 살아납니다. clipLimit이 과한 증폭을 막아 줘요.",
             "Applying CLAHE only to the LAB lightness channel restores local contrast and keeps colors. clipLimit limits over-amplification.", "image"),
            ("cv2.merge([cv2.equalizeHist(c) for c in cv2.split(frame)])", s_eq_bgr, False,
             "BGR 채널을 따로 평활화하면 채널마다 대비가 달라져 색이 틀어집니다. 그래서 밝기(L)만 처리해요.",
             "Equalizing B, G, R separately shifts the colors. That is why only lightness (L) is processed.", "image"),
            ("cv2.GaussianBlur(frame, (11, 11), 0)", s_gauss11, False,
             "블러는 흐리게 할 뿐 밝기 분포(대비)를 넓히지 못합니다.",
             "Blur does not widen the brightness distribution.", "image"),
            ("cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)", s_bgr2gray, False,
             "흑백이 될 뿐 어두운 건 그대로예요.",
             "It becomes gray but stays dark.", "image"),
            ("cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)", s_bgr2hsv, False,
             "HSV 값을 BGR처럼 그리면 색만 엉뚱해집니다.",
             "Drawing HSV as BGR just distorts colors.", "image"),
        ]))
    return qs


# [단계 미션의 구조]  미션 = 여러 "단계". 각 단계의 보기 함수는 fn(st, ctx) -> st' 이다.
#   st   영상과 중간 결과를 담은 상태 dict
#          raw   카메라 원본                 frame  지금까지 처리된 영상 (= 다음 단계의 입력)
#          mask  마스크                      vis    화면에 보여 줄 그림 (없으면 frame)
#          meas  추적 대상 좌표 (x, y)        hand   MediaPipe 랜드마크
#          pts   픽셀 좌표로 바꾼 손 랜드마크
#   ctx  단계마다 따로 있고 프레임이 바뀌어도 유지되는 dict (Kalman 필터처럼 "상태가 있는" 처리용)
# S(st, 키=값) 은 st 를 복사해서 일부 키만 바꾼 새 dict 를 돌려준다 (원본 st 는 건드리지 않음).
def S(st, **kw):
    d = dict(st)
    d.update(kw)
    return d


# 미션이 시작할 때의 상태. raw = 카메라 원본, frame = 망가뜨린 영상. 나머지는 단계가 진행되며 채워진다.
def st_new(raw, frame):
    return dict(raw=raw, frame=frame, mask=None, vis=None, meas=None, hand=None, pts=None)


# 상태에서 "화면에 보여 줄 그림"을 꺼낸다. vis 가 있으면 vis, 없으면 frame. 항상 640x480 BGR 로
# 통일한다.
def vis_img(st):
    v = st.get("vis")
    return to_view(v if v is not None else st["frame"])


# 보기 함수(s_*)를 단계 함수 fn(st, ctx) 로 감싸는 어댑터 4종. 차이는 "무엇을 바꾸느냐"다.
#   f_frame   결과로 frame 자체를 교체 -> 다음 단계의 입력이 실제로 바뀐다 (뒤집기, 노이즈 제거 등)
def f_frame(fn):
    def run(st, ctx):
        return S(st, frame=to_view(fn(st["frame"])), vis=None)
    return run


#   f_view    화면(vis)에만 결과를 보여 주고 frame 은 그대로 둔다 (윤곽선, 코너처럼 "보기용" 결과)
def f_view(fn):
    def run(st, ctx):
        return S(st, vis=to_view(fn(st["frame"])))
    return run


# 마스크에 흰 점 노이즈를 일부러 섞는다. 그래서 미션 2 의 3단계(OPEN)가 꼭 필요한 문제가 된다.
def with_specks(m):
    pool = _speck_pool()
    return cv2.bitwise_or(m, pool[random.randrange(len(pool))])


#   f_mask    frame 에서 마스크를 만들어 st["mask"] 에 저장 (점 노이즈 포함)
def f_mask(fn):
    def run(st, ctx):
        m = with_specks(fn(st["frame"]))
        return S(st, mask=m, vis=to_view(m))
    return run


#   f_maskop  이미 있는 st["mask"] 에 모폴로지 같은 연산을 적용해서 mask 를 갱신
def f_maskop(fn):
    def run(st, ctx):
        m = fn(st["mask"])
        return S(st, mask=m, vis=to_view(m))
    return run


# cv2.findContours 의 반환값이 OpenCV 3.x 는 (영상, 윤곽, 계층), 4.x 이상은 (윤곽, 계층) 이다. 길이로
# 구분해서 둘 다 처리한다. 입력은 반드시 1채널 마스크여야 한다.
def _contours(mask):
    res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(res[0] if len(res) == 2 else res[1])


# 미션 4 의 추적 대상: 피부색 마스크를 OPEN 으로 정리한 뒤 가장 큰 윤곽. 면적이 400 미만이면 없는 것으로
# 본다.
def _skin_blob(frame):
    m = cv2.morphologyEx(skin_mask(frame), cv2.MORPH_OPEN, _K5)
    cs = _contours(m)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    return c if cv2.contourArea(c) >= 400 else None


# cv2.circle 등에 넘기는 좌표는 정수 범위가 제한돼 있다. 잘못된 보기(예: m10 을 면적으로 나누지 않음)가
# 만드는 엄청 큰 값에서도 그리기 함수가 에러 나지 않도록 +-30000 으로 잘라 정수로 만든다.
def _ci(v):
    return int(max(-30000, min(30000, v)))


def s_rot180(f): return cv2.rotate(f, cv2.ROTATE_180)
def s_chan0(f): return f[:, :, 0].copy()
def s_dim(f): return cv2.convertScaleAbs(f, alpha=0.5)
def s_canny_lo(f): return cv2.Canny(_gray_of(f), 10, 30)
def s_canny_hi(f): return cv2.Canny(_gray_of(f), 240, 255)
def s_erode4(m): return cv2.erode(m, _K5, iterations=4)


def s_skin_wide(f):
    return cv2.inRange(cv2.cvtColor(f, cv2.COLOR_BGR2HSV),
                       np.array([0, 40, 70], np.uint8), np.array([180, 255, 255], np.uint8))


# Harris 코너: 흑백 float32 -> cornerHarris -> dilate 로 점을 키우고, 응답이 (최댓값 x ratio) 보다 큰
# 곳을 빨갛게 칠한다. ratio 가 너무 작으면(0.0001) 거의 모든 곳이 코너가 되고, 너무 크면(0.6) 거의
# 사라진다. 정답은 0.01.
def _harris(frame, ratio):
    g32 = np.float32(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    dst = cv2.dilate(cv2.cornerHarris(g32, 2, 3, 0.04), None)
    out = frame.copy()
    if dst.max() > 0:
        out[dst > ratio * dst.max()] = (0, 0, 255)
    return out


def s_harris_ok(f): return _harris(f, 0.01)
def s_harris_low(f): return _harris(f, 0.0001)
def s_harris_high(f): return _harris(f, 0.6)


def s_harris_color(f):
    return cv2.cornerHarris(f, 2, 3, 0.04)


def _draw_blob_marker(img, c, p, label=True):
    ix, iy = _ci(p[0]), _ci(p[1])
    if 0 <= ix < VID_W and 0 <= iy < VID_H:
        cv2.drawMarker(img, (ix, iy), (0, 0, 255), cv2.MARKER_CROSS, 28, 3)
        cv2.circle(img, (ix, iy), 6, (0, 0, 255), -1)
    if label:
        outlined(img, "(cx, cy) = (%d, %d)%s" % (_ci(p[0]), _ci(p[1]),
                                                 "" if (0 <= ix < VID_W and 0 <= iy < VID_H) else "  OFF SCREEN"),
                 (10, 70), 0.5, C_WARN, 1)


# 미션 2 의 4단계 정답: 마스크에서 면적이 가장 큰 윤곽을 골라 초록 테두리를 그리고, moments 로 중심점을
# 구해 십자로 표시한다. 중심 = (m10 / m00, m01 / m00)  (m00 은 면적).
def c_largest(st, ctx):
    img = st["frame"].copy()
    cs = _contours(st["mask"])
    if cs:
        c = max(cs, key=cv2.contourArea)
        cv2.drawContours(img, [c], -1, (0, 255, 0), 3)
        M = cv2.moments(c)
        if M["m00"] > 0:
            _draw_blob_marker(img, c, (M["m10"] / M["m00"], M["m01"] / M["m00"]))
    else:
        outlined(img, "no contour - show your face/hand", (10, 70), 0.5, C_WARN, 1)
    return S(st, vis=img)


def c_first(st, ctx):
    img = st["frame"].copy()
    cs = _contours(st["mask"])
    if cs:
        c = cs[0]
        cv2.drawContours(img, [c], -1, (0, 255, 0), 3)
        M = cv2.moments(c)
        if M["m00"] > 0:
            _draw_blob_marker(img, c, (M["m10"] / M["m00"], M["m01"] / M["m00"]))
    return S(st, vis=img)


def c_all(st, ctx):
    img = st["frame"].copy()
    cv2.drawContours(img, _contours(st["mask"]), -1, (0, 255, 0), 2)
    return S(st, vis=img)


# 함정: findContours 에 3채널 컬러 frame 을 넣으면 에러가 난다 (마스크 1채널이어야 함).
def c_on_frame(st, ctx):
    _contours(st["frame"])
    return st


def c_bbox(st, ctx):
    img = st["frame"].copy()
    x, y, w, h = cv2.boundingRect(st["mask"])
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return S(st, vis=img)


# 미션 4 의 1단계. variant(윤곽) -> (x, y) 가 보기마다 다르다. 정답은 _cen_ok (m10/m00, m01/m00).
# 계산한 점을 st["meas"] 로 넘기면 2단계의 Kalman 필터가 그 점을 측정값으로 쓴다.
def make_centroid(variant):
    def run(st, ctx):
        img = st["frame"].copy()
        c = _skin_blob(st["frame"])
        if c is None:
            outlined(img, "NO TARGET - show your face or hand", (10, 70), 0.5, C_WARN, 1)
            return S(st, vis=img, meas=None)
        cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
        p = variant(c)
        _draw_blob_marker(img, c, p)
        return S(st, vis=img, meas=(float(p[0]), float(p[1])))
    return run


def _cen_ok(c):
    M = cv2.moments(c)
    return M["m10"] / M["m00"], M["m01"] / M["m00"]


def _cen_swap(c):
    M = cv2.moments(c)
    return M["m01"] / M["m00"], M["m10"] / M["m00"]


def _cen_raw(c):
    M = cv2.moments(c)
    return M["m10"], M["m01"]


def _cen_corner(c):
    x, y, w, h = cv2.boundingRect(c)
    return x, y


def _cen_first(c):
    return float(c[0][0][0]), float(c[0][0][1])


# Kalman 전이 행렬 후보. 상태는 [x, y, vx, vy].
#   KF_CV   정답. 다음 x = x + vx, 다음 y = y + vy  (등속도 모델)
#   KF_NEG  속도 부호가 반대            KF_ID  단위행렬이라 속도가 위치에 반영되지 않음
# KF_AHEAD = 노란 화살표가 몇 프레임 앞을 가리키는지.
# LOSS_PERIOD / LOSS_LEN = 3초마다 1.2초 동안 측정값이 사라지는 "가려짐" 흉내.
KF_CV = [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]
KF_NEG = [[1, 0, -1, 0], [0, 1, 0, -1], [0, 0, 1, 0], [0, 0, 0, 1]]
KF_ID = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
KF_AHEAD = 8
LOSS_PERIOD, LOSS_LEN = 3.0, 1.2


# cv2.KalmanFilter(4, 2) 를 만든다: 상태 4개 [x, y, vx, vy], 측정 2개 [x, y].
#   transitionMatrix    F  상태가 한 프레임에 어떻게 변하는가
#   measurementMatrix   H  상태 중 무엇을 측정하는가 (위치 x, y 만)
#   processNoiseCov     Q  모델이 틀릴 수 있는 정도 (클수록 모델을 덜 믿음)
#   measurementNoiseCov R  측정값이 흔들리는 정도 (클수록 측정을 덜 믿음)
#   errorCovPost        P  처음 상태에 대한 불확실성 (크게 줘서 첫 측정에 빨리 맞추게 함)
# Q, R, P 의 숫자는 동작을 보면서 정한 초기값이지 이론으로 유도한 값이 아니다.
# 행렬은 반드시 float32 numpy 배열이어야 한다 (리스트나 float64 는 에러).
def _make_kf(cfg, x, y):
    if cfg.get("dim", 4) == 4:
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array(cfg.get("trans", KF_CV), np.float32)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.processNoiseCov = np.diag([1e-2, 1e-2, 5e-2, 5e-2]).astype(np.float32)
        kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0
        kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
    else:
        kf = cv2.KalmanFilter(2, 2)
        kf.transitionMatrix = np.eye(2, dtype=np.float32)
        kf.measurementMatrix = np.eye(2, dtype=np.float32)
        kf.processNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
        kf.errorCovPost = np.eye(2, dtype=np.float32) * 10.0
        kf.statePost = np.array([[x], [y]], np.float32)
    kf.statePre = kf.statePost.copy()
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * float(cfg.get("r", 4.0))
    return kf


# Kalman 한 프레임 처리.  pred = kf.predict() 를 부른 뒤
#   측정이 있으면  est = kf.correct(z)   /   없으면  pred 를 추정으로 사용 (lost_mode 에 따라 다름)
# ctx 에 필터(kf), 마지막 측정(last), 놓친 프레임 수(miss), 궤적(trail)을 저장해 프레임 사이에 이어
# 간다. lost_mode ("측정이 사라졌을 때" 처리 방식):
#   predict  예측만 사용 (정답)       hold    옛 측정으로 계속 보정      zeros   0 을 측정값으로 보정
#   origin   추정을 0 으로            reset   필터를 새로 만듦
# simulate_loss=True 이면 3초마다 1.2초간 측정을 지우고 화면을 어둡게 해서 가려짐을 흉내 낸다.
# 25프레임 넘게 못 보면 추적을 포기하고 필터를 지운다 (가려짐 흉내 중에는 제외).
# 그림: 빨간 원 = 측정, 초록 점 = 추정, 노란 화살표 = KF_AHEAD 프레임 뒤 예상 위치, 하늘색 선 = 궤적.
def kalman_frame(st, ctx, cfg, lost_mode="predict", simulate_loss=False):
    img = st["frame"].copy()
    now = ctx.get("now", 0.0)
    t0 = ctx.setdefault("t0", now)
    meas = st.get("meas")
    lost = simulate_loss and ((now - t0) % LOSS_PERIOD) >= (LOSS_PERIOD - LOSS_LEN)
    if lost:
        meas = None
        img = (img.astype(np.float32) * 0.45).astype(np.uint8)
        outlined(img, "TARGET LOST (simulated occlusion)", (20, 200), 0.7, C_BAD, 2)
    kf = ctx.get("kf")
    if kf is None and meas is not None:
        kf = ctx["kf"] = _make_kf(cfg, meas[0], meas[1])
        ctx["last"] = meas
        ctx["miss"] = 0
        ctx["trail"] = deque(maxlen=40)
    est = None
    if kf is not None:
        pred = kf.predict()
        if meas is not None:
            est = kf.correct(np.array([[meas[0]], [meas[1]]], np.float32))
            ctx["last"] = meas
            ctx["miss"] = 0
        else:
            ctx["miss"] = ctx.get("miss", 0) + 1
            if lost_mode == "predict":
                est = pred
            elif lost_mode == "hold":
                est = kf.correct(np.array([[ctx["last"][0]], [ctx["last"][1]]], np.float32))
            elif lost_mode == "zeros":
                est = kf.correct(np.zeros((2, 1), np.float32))
            elif lost_mode == "origin":
                est = np.zeros_like(pred)
            elif lost_mode == "reset":
                ctx["kf"] = None
            if not lost and ctx["miss"] > 25:
                ctx["kf"] = None
                est = None
    if meas is not None:
        cv2.circle(img, (_ci(meas[0]), _ci(meas[1])), 9, (0, 0, 255), 2)
    if est is not None:
        v = np.asarray(est, np.float64).flatten()
        ex, ey = float(v[0]), float(v[1])
        vx, vy = (float(v[2]), float(v[3])) if len(v) >= 4 else (0.0, 0.0)
        tr = ctx.setdefault("trail", deque(maxlen=40))
        tr.append((_ci(ex), _ci(ey)))
        if len(tr) > 1:
            cv2.polylines(img, [np.array(tr, np.int32)], False, (255, 255, 0), 2)
        cv2.circle(img, (_ci(ex), _ci(ey)), 7, (0, 255, 0), -1)
        cv2.arrowedLine(img, (_ci(ex), _ci(ey)), (_ci(ex + vx * KF_AHEAD), _ci(ey + vy * KF_AHEAD)),
                        (0, 255, 255), 3, tipLength=0.25)
    elif meas is None and not lost:
        outlined(img, "NO TARGET - show your face or hand", (10, 90), 0.5, C_WARN, 1)
    outlined(img, "red=measure  green=estimate  yellow=%d frames ahead" % KF_AHEAD, (10, 70), 0.45, C_TEXT, 1)
    return S(st, vis=img)


def make_kalman(cfg, lost_mode="predict", simulate_loss=False):
    def run(st, ctx):
        return kalman_frame(st, ctx, cfg, lost_mode, simulate_loss)
    return run


# h_detect 는 모듈 수준 함수라 Quiz 객체에 접근할 수 없다. 그래서 Quiz.__init__ 이 이 dict 에
# HandTracker 를 넣어 두고, h_detect 가 꺼내 쓴다 (모델은 프로그램 전체에서 한 번만 로드된다).
_HANDS = {"tracker": None}


# 손 뼈대 그리기: HAND_CONNECTIONS 의 쌍마다 초록 선, 21개 점은 파란 점(손가락 끝 tip 은 더 큰 빨간 점).
def draw_hand(img, pts):
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], (0, 255, 0), 2)
    for i, pt in enumerate(pts):
        tip = i in (4, 8, 12, 16, 20)
        cv2.circle(img, pt, 5 if tip else 3, (0, 0, 255) if tip else (255, 0, 0), -1)


# 미션 5 의 1단계 정답. 프레임을 MediaPipe 에 넣어 랜드마크를 구해 st["hand"] 에 저장한다.
# 여기서는 "몇 개 검출됐는지" 글씨만 그리고 뼈대는 그리지 않는다. 뼈대는 2단계(make_pix)에서 좌표를
# 픽셀로 바꾼 뒤 그려진다. HandTracker.detect() 안에서 BGR -> RGB 변환을 하므로, 이 보기를 고르는 것이
# "RGB 로 바꿔서 넣기"에 해당한다.
def h_detect(st, ctx):
    t = _HANDS["tracker"]
    lms = t.detect(st["frame"]) if t is not None else []
    img = st["frame"].copy()
    if lms:
        outlined(img, "MediaPipe: %d hand(s), 21 landmarks each (x, y in 0~1)" % len(lms), (10, 70), 0.5, C_GOOD, 1)
    else:
        outlined(img, "No hand detected - show your hand to the camera", (10, 70), 0.5, C_WARN, 1)
    return S(st, hand=lms, vis=img)


# 오답 보기용: 모델에 잘못 들어갈 입력(R/B 뒤바뀜, 1채널, HSV, 64x48)이 어떤 모습인지 영상으로 보여
# 준다. 실제로 MediaPipe 를 다시 돌리지는 않고 hand 를 빈 목록으로 둔다.
def h_model_view(fn, note):
    def run(st, ctx):
        v = to_view(fn(st["frame"])).copy()
        outlined(v, note, (10, 70), 0.5, C_WARN, 1)
        return S(st, vis=v, hand=[])
    return run


# 정규화 좌표(0~1) -> 픽셀 좌표 변환 후보. 정답은 _pix_ok: x 에 폭(w), y 에 높이(h)를 곱한다.
# 나머지는 곱하지 않음 / w, h 를 바꿔 곱함 / x, y 를 바꿈 / y 를 1 에서 뺌(위아래 뒤집힘)이라는 흔한
# 실수들이다.
def _pix_ok(p, w, h): return int(p.x * w), int(p.y * h)
def _pix_noscale(p, w, h): return int(p.x), int(p.y)
def _pix_hw(p, w, h): return int(p.x * h), int(p.y * w)
def _pix_swap(p, w, h): return int(p.y * w), int(p.x * h)
def _pix_flip(p, w, h): return int(p.x * w), int((1 - p.y) * h)


# 미션 5 의 2단계. 21개 랜드마크를 variant 로 픽셀 좌표로 바꿔 st["pts"] 에 저장하고 뼈대를 그린다.
# 오답이면 뼈대가 손 위가 아닌 엉뚱한 곳에 그려져서 틀린 이유가 눈에 보인다.
def make_pix(variant):
    def run(st, ctx):
        img = st["frame"].copy()
        h, w = img.shape[:2]
        hands = st.get("hand") or []
        allpts = []
        for lm in hands:
            pts = [variant(p, w, h) for p in lm]
            draw_hand(img, pts)
            allpts.append(pts)
        if not hands:
            outlined(img, "No hand detected - show your hand to the camera", (10, 70), 0.5, C_WARN, 1)
        return S(st, vis=img, pts=allpts)
    return run


# 미션 5 의 3단계. 손가락 5개마다 관절 각도(calculate_angle)를 구하고 pred(각도) 가 참이면 "펴짐".
# 펴진 손가락은 초록, 접힌 손가락은 빨간 원으로 끝에 표시하고 손목 아래에 "Fingers: n" 을 쓴다.
# 정답은 각도 >= 160. 부등호가 반대이거나 기준이 너무 낮거나(20) 높으면(179) 개수가 엉터리로 나온다.
def make_rule(pred):
    def run(st, ctx):
        img = st["frame"].copy()
        hands = st.get("hand") or []
        allp = st.get("pts") or []
        for lm, pts in zip(hands, allp):
            draw_hand(img, pts)
            n = 0
            for (a, b, c), tip in zip(FINGER_ANGLE_POINTS, (4, 8, 12, 16, 20)):
                ext = bool(pred(calculate_angle(lm[a], lm[b], lm[c])))
                n += ext
                cv2.circle(img, pts[tip], 12, (0, 255, 0) if ext else (0, 0, 255), 3)
            outlined(img, "Fingers: %d" % n, (pts[0][0] - 30, pts[0][1] + 34), 0.9, C_WARN, 2)
        if not hands:
            outlined(img, "No hand detected - show your hand to the camera", (10, 70), 0.5, C_WARN, 1)
        return S(st, vis=img)
    return run


# 미션 보기는 항상 kind="st" 인 6-튜플이다. 5-튜플로 짧게 적을 수 있게 해 주는 도우미.
def ch(code, fn, ok, ko, en):
    return (code, fn, ok, ko, en, "st")


# 미션 5개의 데이터. 미션 하나 = dict:
#   id, name_ko, name_en   번호와 이름
#   needs_hands            True 면 MediaPipe + 모델 파일이 있어야 선택 가능 (미션 5)
#   source(raw) -> st      시작 상태. 카메라 원본을 일부러 망가뜨린 상태 (예: 뒤집기 + 노이즈)
#   stages                 단계 목록. 단계 하나 = dict:
#                            short_ko / short_en, unit_name   패널 칩과 헤더에 쓰는 이름
#                            ko / en                          단계 문제 문장
#                            setup, metric                    문제 아래 코드 줄들, 측정 숫자
#                            choices                          ch() 로 만든 보기 5개
# 이 함수의 맨 아래에서 단계마다 _correct_fn (정답 보기 함수)을,
# 미션마다 final_lines (완성 코드)를 추가한다.
def build_missions():
    sharp_setup = "# sharp_kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]], np.float32)"
    ms = []

    ms.append(dict(
        id=1, name_ko="뒤집힘 + 노이즈 → 윤곽선", name_en="Flip + noise -> edges", needs_hands=False,
        # 미션 1 의 시작 상태: 뒤집고(p_flip_ud) 노이즈를 더한(p_noise) 영상. 1단계 정답이 뒤집기, 2단계
        # 정답이 노이즈 제거다.
        source=lambda raw: st_new(raw, p_noise(p_flip_ud(raw))),
        stages=[
            dict(short_ko="뒤집기", short_en="Flip", unit_name="Flip",
                 ko="1단계: 카메라가 위아래로 뒤집혀 있고 노이즈도 낍니다. 먼저 화면을 바로 세우세요.",
                 en="Step 1: the camera is upside down and noisy. First turn the view upright.",
                 setup=[("# frame : 위아래가 뒤집힌 영상 (노이즈 포함)", "# frame : upside-down image (with noise)")], metric=None,
                 choices=[
                     ch("frame = cv2.flip(frame, 0)", f_frame(s_flip0), True,
                        "flipCode 0은 상하 반전입니다. 화면이 바로 섰고, 이 결과가 다음 단계의 입력이 됩니다.",
                        "flipCode 0 = vertical flip. The view is upright and feeds the next step."),
                     ch("frame = cv2.flip(frame, 1)", f_frame(s_flip1), False,
                        "좌우 반전이라 위아래는 그대로 뒤집혀 있어요.",
                        "That mirrors left-right; it stays upside down."),
                     ch("frame = cv2.flip(frame, -1)", f_frame(s_flipm1), False,
                        "상하와 좌우를 모두 뒤집어서 결국 좌우만 뒤집힌 상태가 됩니다.",
                        "-1 flips both axes, leaving a left-right mirrored view."),
                     ch("frame = frame[:, ::-1]", f_frame(s_slice_lr), False,
                        "열 순서를 뒤집는 NumPy 좌우 반전입니다. 위아래는 그대로예요.",
                        "Reverses the columns (left-right). Still upside down."),
                     ch("frame = cv2.rotate(frame, cv2.ROTATE_180)", f_frame(s_rot180), False,
                        "180도 회전은 상하+좌우 반전과 같아서 좌우만 뒤집힌 화면이 됩니다.",
                        "A 180-degree rotation equals flipping both axes: left-right mirrored."),
                 ]),
            dict(short_ko="노이즈 제거", short_en="Denoise", unit_name="Denoise",
                 ko="2단계: 이제 자글자글한 노이즈를 지우세요. 윤곽선을 뽑기 전에 꼭 필요한 전처리입니다.",
                 en="Step 2: remove the grainy noise. Required preprocessing before edge detection.",
                 setup=[("# frame : 바로 선 영상 (노이즈는 그대로)", "# frame : upright image (still noisy)"), sharp_setup], metric="noise",
                 choices=[
                     ch("frame = cv2.GaussianBlur(frame, (9, 9), 0)", f_frame(s_gauss9), True,
                        "가우시안 가중 평균으로 랜덤 노이즈가 크게 줄었습니다. 노이즈 수치가 떨어진 걸 보세요.",
                        "A Gaussian average removed most of the random noise. Watch the number drop."),
                     ch("frame = cv2.filter2D(frame, -1, sharp_kernel)", f_frame(s_sharp), False,
                        "샤프닝은 픽셀 차이를 키워서 노이즈도 같이 증폭합니다.",
                        "Sharpening boosts pixel differences, so the noise grows too."),
                     ch("frame = cv2.GaussianBlur(frame, (1, 1), 0)", f_frame(s_gauss1), False,
                        "커널 크기 1x1은 주변 픽셀을 하나도 섞지 않아서 화면이 그대로입니다. 커널이 너무 작으면 효과가 없어요.",
                        "A 1x1 kernel mixes no neighbors, so the view stays the same. Too small a kernel does nothing."),
                     ch("frame = cv2.Canny(frame, 100, 200)", f_frame(s_canny_bgr), False,
                        "노이즈 점마다 엣지로 잡혀서 더 지저분해집니다. 노이즈 제거가 먼저예요.",
                        "Every noisy pixel becomes an edge. Denoise first."),
                     ch("frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)", f_frame(s_bgr2hsv), False,
                        "색공간만 바뀌고 노이즈는 그대로입니다.",
                        "Only the color space changes; the noise stays."),
                 ]),
            dict(short_ko="윤곽선", short_en="Edges", unit_name="Edges",
                 ko="3단계: 깨끗해진 영상에서 윤곽선(엣지)만 뽑아내세요.",
                 en="Step 3: extract only the edges from the clean image.",
                 setup=[("# frame : 바로 서고 깨끗해진 영상", "# frame : upright and clean image"),
                        "# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)"], metric=None,
                 choices=[
                     ch("edges = cv2.Canny(gray, 100, 200)", f_view(s_canny_gray), True,
                        "Canny는 uint8 흑백 영상용입니다. 노이즈를 먼저 지웠기 때문에 엣지가 깔끔하게 나와요.",
                        "Canny works on uint8 grayscale. Because we denoised first, the edges are clean."),
                     ch("edges = cv2.Canny(gray, 10, 30)", f_view(s_canny_lo), False,
                        "임계값이 너무 낮아서 약한 무늬까지 전부 엣지가 됩니다.",
                        "The thresholds are too low; even faint texture becomes an edge."),
                     ch("edges = cv2.Canny(gray, 240, 255)", f_view(s_canny_hi), False,
                        "임계값이 너무 높아서 거의 아무 엣지도 남지 않습니다.",
                        "The thresholds are too high; almost nothing survives."),
                     ch("edges = cv2.Canny(gray.astype(np.float32) / 255., 100, 200)", f_view(s_canny_f32), False,
                        "Canny는 uint8 전용입니다. float32로 바꾸면 에러가 나요.",
                        "Canny only takes uint8. float32 input raises an error."),
                     ch("edges = cv2.GaussianBlur(gray, (11, 11), 0)", f_view(s_blur_gray), False,
                        "블러는 경계를 뭉개서 엣지를 만들지 못합니다.",
                        "Blur smears boundaries; it does not extract edges."),
                 ]),
        ]))

    ms.append(dict(
        id=2, name_ko="색 바로잡기 → 피부색 → 윤곽", name_en="Color -> skin -> contour", needs_hands=False,
        source=lambda raw: st_new(raw, p_swap_rb(raw)),
        stages=[
            dict(short_ko="색 바로잡기", short_en="Color fix", unit_name="Color fix",
                 ko="1단계: 피부가 파랗게 보입니다(B와 R 채널이 뒤바뀜). 색부터 바로잡으세요.",
                 en="Step 1: skin looks blue (B and R swapped). Fix the colors first.",
                 setup=[("# frame : B와 R 채널이 뒤바뀐 영상", "# frame : image with B and R swapped")], metric=None,
                 choices=[
                     ch("frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)", f_frame(s_bgr2rgb), True,
                        "B와 R을 맞바꿔서 원래 색으로 돌아왔습니다. 이제 HSV 색 판단이 제대로 동작해요.",
                        "Swapping B and R restores the colors, so HSV color tests work again."),
                     ch("frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)", f_frame(s_bgr2gray), False,
                        "흑백이 될 뿐 색을 바로잡은 게 아닙니다. 피부색을 찾을 수 없게 돼요.",
                        "Grayscale discards color; you could no longer find skin."),
                     ch("frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)", f_frame(s_bgr2hsv), False,
                        "HSV 값을 BGR처럼 그리면 색이 엉뚱해집니다.",
                        "HSV values drawn as BGR give wrong colors."),
                     ch("frame = frame[:, :, 0]", f_frame(s_chan0), False,
                        "채널 하나만 꺼내면 흑백(파랑 채널)이 됩니다.",
                        "Taking a single channel gives a gray image."),
                     ch("frame = cv2.flip(frame, 1)", f_frame(s_flip1), False,
                        "좌우 반전은 색과 무관합니다.",
                        "A flip has nothing to do with color."),
                 ]),
            dict(short_ko="피부색 마스크", short_en="Skin mask", unit_name="Skin mask",
                 ko="2단계: 피부색(얼굴·손)만 남기는 마스크를 만드세요. 카메라에 얼굴이나 손을 보여 주세요.",
                 en="Step 2: build a mask that keeps only skin color. Show your face or hand.",
                 setup=["# hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)", "# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)",
                        ("# 카메라 노이즈 때문에 마스크에 작은 흰 점이 섞입니다", "# camera noise leaves small white specks in the mask")], metric=None,
                 choices=[
                     ch("mask = cv2.inRange(hsv, (0, 40, 70), (20, 255, 255))", f_mask(s_skin_hsv), True,
                        "색 판단은 HSV에서 합니다. 피부색은 H 0~20 부근이라 얼굴과 손이 흰색으로 남았어요. 작은 점 노이즈는 다음 단계에서 지웁니다.",
                        "Judge color in HSV; skin hue is about 0-20. Specks are removed in the next step."),
                     ch("mask = cv2.inRange(frame, (0, 40, 70), (20, 255, 255))", f_mask(s_skin_bgr), False,
                        "BGR 값에 HSV 범위를 쓰면 엉뚱한 픽셀이 선택됩니다. 먼저 HSV로 변환해야 해요.",
                        "Using HSV ranges on BGR values selects the wrong pixels."),
                     ch("mask = cv2.inRange(hsv, (100, 40, 70), (130, 255, 255))", f_mask(s_blue_hsv), False,
                        "H 100~130은 파랑 계열입니다. 피부색이 아니에요.",
                        "H 100-130 is blue, not skin."),
                     ch("mask = cv2.inRange(hsv, (0, 40, 70), (180, 255, 255))", f_mask(s_skin_wide), False,
                        "H 범위를 0~180으로 넓히면 거의 모든 색이 통과해서 화면 대부분이 흰색이 됩니다.",
                        "H 0-180 lets almost every hue through; most of the screen turns white."),
                     ch("mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]", f_mask(s_thresh_gray), False,
                        "색이 아니라 밝기로 나누기 때문에 밝은 영역이 전부 남습니다.",
                        "That splits by brightness, not by color."),
                 ]),
            dict(short_ko="점 지우기", short_en="Open", unit_name="Open",
                 ko="3단계: 마스크의 작은 흰 점은 지우고 큰 덩어리(얼굴·손)는 살리세요.",
                 en="Step 3: remove the tiny white specks but keep the big blob.",
                 setup=["# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))",
                        ("# mask : 피부색 마스크 + 작은 흰 점 노이즈", "# mask : skin mask + small white specks")],
                 metric=None,
                 choices=[
                     ch("mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)", f_maskop(s_open), True,
                        "Open = 침식 후 팽창. 점은 침식에서 사라지고 큰 덩어리는 팽창으로 원래 크기를 되찾습니다.",
                        "Open = erode then dilate. Specks vanish; the big blob grows back."),
                     ch("mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)", f_maskop(s_close), False,
                        "Close는 검은 구멍을 메웁니다. 흰 점은 그대로 남아요.",
                        "Close fills black holes; white specks stay."),
                     ch("mask = cv2.dilate(mask, kernel)", f_maskop(s_dilate), False,
                        "팽창은 흰 영역을 키우므로 점이 더 커집니다.",
                        "Dilation grows white areas, so specks get bigger."),
                     ch("mask = cv2.erode(mask, kernel, iterations=4)", f_maskop(s_erode4), False,
                        "침식만 하면 점은 사라지지만 큰 덩어리도 같이 쪼그라듭니다. Open은 팽창으로 크기를 되돌려요.",
                        "Erosion alone removes specks but also shrinks the blob. Open dilates it back."),
                     ch("mask = cv2.GaussianBlur(mask, (5, 5), 0)", f_maskop(s_gauss_mask), False,
                        "블러는 점을 회색 얼룩으로 번지게 할 뿐 지우지 못합니다.",
                        "Blur smears specks into gray blotches instead of removing them."),
                 ]),
            dict(short_ko="윤곽·중심", short_en="Contour", unit_name="Contour",
                 ko="4단계: 마스크에서 가장 큰 윤곽(얼굴·손)을 찾아 테두리와 중심점을 표시하세요.",
                 en="Step 4: find the largest contour in the mask and mark its outline and center.",
                 setup=["# contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)"], metric=None,
                 choices=[
                     ch("cnt = max(contours, key=cv2.contourArea)", c_largest, True,
                        "면적이 가장 큰 윤곽을 고르고, moments로 중심점(빨간 십자)까지 구했습니다. 작은 잔여 점은 무시돼요.",
                        "The largest contour by area, plus its center via moments. Leftover specks are ignored."),
                     ch("cnt = contours[0]", c_first, False,
                        "contours[0]은 그냥 처음 발견된 윤곽입니다. 큰 덩어리가 아니라 작은 점일 수 있어요.",
                        "contours[0] is simply the first one found; it may be a tiny speck."),
                     ch("cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)", c_all, False,
                        "모든 윤곽을 그리면 남은 작은 점들까지 전부 표시돼서 어수선합니다.",
                        "Drawing every contour also outlines the leftover specks."),
                     ch("contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)", c_on_frame, False,
                        "findContours는 3채널 컬러가 아니라 단일 채널 마스크를 받습니다. 그래서 에러가 나요.",
                        "findContours needs a single-channel mask, not a color frame, so it raises an error."),
                     ch("x, y, w, h = cv2.boundingRect(mask)", c_bbox, False,
                        "마스크 전체를 감싸는 사각형이라 멀리 남은 점까지 포함해서 엉뚱하게 커집니다.",
                        "That box wraps the whole mask, including far-away specks."),
                 ]),
        ]))

    ms.append(dict(
        id=3, name_ko="어두운 화면 → 선명 → 코너", name_en="Dark -> sharp -> corners", needs_hands=False,
        source=lambda raw: st_new(raw, p_dark(p_blur(raw))),
        stages=[
            dict(short_ko="밝게", short_en="CLAHE", unit_name="CLAHE",
                 ko="1단계: 화면이 어둡고 흐립니다. 색은 유지하면서 대비부터 살리세요.",
                 en="Step 1: the view is dark and blurry. Restore contrast first, keeping colors.",
                 setup=["# clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))",
                        "# L, a, b = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))"], metric="contrast",
                 choices=[
                     ch("frame = cv2.cvtColor(cv2.merge([clahe.apply(L), a, b]), cv2.COLOR_LAB2BGR)", f_frame(clahe_on_L), True,
                        "LAB의 밝기(L)에만 CLAHE를 적용해서 색은 유지한 채 구간별 대비가 살아났습니다.",
                        "CLAHE on the LAB lightness restores local contrast and keeps colors."),
                     ch("frame = cv2.merge([cv2.equalizeHist(c) for c in cv2.split(frame)])", f_frame(s_eq_bgr), False,
                        "채널을 따로 평활화하면 채널마다 대비가 달라져 색이 틀어집니다.",
                        "Equalizing channels separately shifts the colors."),
                     ch("frame = cv2.convertScaleAbs(frame, alpha=0.5)", f_frame(s_dim), False,
                        "alpha가 1보다 작으면 오히려 더 어두워집니다.",
                        "alpha < 1 makes it even darker."),
                     ch("frame = cv2.GaussianBlur(frame, (11, 11), 0)", f_frame(s_gauss11), False,
                        "블러는 대비를 넓히지 못하고 더 흐리게 만듭니다.",
                        "Blur does not widen contrast; it blurs more."),
                     ch("frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)", f_frame(s_bgr2gray), False,
                        "흑백이 될 뿐 어두운 건 그대로예요.",
                        "It becomes gray but stays dark."),
                 ]),
            dict(short_ko="선명하게", short_en="Sharpen", unit_name="Sharpen",
                 ko="2단계: 밝아졌지만 여전히 흐립니다. 선명하게 만드세요.",
                 en="Step 2: brighter, but still blurry. Sharpen it.",
                 setup=[sharp_setup], metric="sharp",
                 choices=[
                     ch("frame = cv2.filter2D(frame, -1, sharp_kernel)", f_frame(s_sharp), True,
                        "중심 9, 이웃 -1 커널로 엣지가 강조되어 선명해졌습니다. 선명도 숫자가 크게 올랐어요.",
                        "Center 9, neighbors -1: edges are boosted. The sharpness number jumps."),
                     ch("frame = cv2.GaussianBlur(frame, (21, 21), 0)", f_frame(s_gauss21), False,
                        "블러라서 훨씬 더 흐려집니다.",
                        "A blur makes it much blurrier."),
                     ch("frame = cv2.blur(frame, (21, 21))", f_frame(s_box21), False,
                        "Box 블러도 평균이라 더 흐려집니다.",
                        "A box blur is an average too; blurrier."),
                     ch("frame = cv2.Canny(frame, 100, 200)", f_frame(s_canny_bgr), False,
                        "윤곽선만 남기는 함수라 선명해지는 게 아닙니다.",
                        "Canny keeps outlines only; it does not sharpen."),
                     ch("frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)", f_frame(s_bgr2gray), False,
                        "흑백으로 바뀔 뿐 선명도는 그대로입니다.",
                        "Only gray; sharpness unchanged."),
                 ]),
            dict(short_ko="코너", short_en="Corners", unit_name="Harris",
                 ko="3단계: 선명해진 영상에서 모서리(코너)를 찾아 빨갛게 표시하세요.",
                 en="Step 3: find the corners in the sharp image and mark them red.",
                 setup=["# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)",
                        "# dst = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)",
                        "# dst = cv2.dilate(dst, None)   # make the corner blobs bigger"],
                 metric=None,
                 choices=[
                     ch("frame[dst > 0.01 * dst.max()] = (0, 0, 255)", f_view(s_harris_ok), True,
                        "Harris 응답의 최댓값 대비 1% 이상인 곳만 코너로 표시했습니다. 적당한 임계값이에요.",
                        "Marking responses above 1% of the maximum gives a reasonable set of corners."),
                     ch("frame[dst > 0.0001 * dst.max()] = (0, 0, 255)", f_view(s_harris_low), False,
                        "임계값이 너무 낮아서 거의 모든 곳이 코너가 됩니다.",
                        "The threshold is too low; almost everything becomes a corner."),
                     ch("frame[dst > 0.6 * dst.max()] = (0, 0, 255)", f_view(s_harris_high), False,
                        "임계값이 너무 높아서 코너가 거의 남지 않습니다.",
                        "The threshold is too high; hardly any corner survives."),
                     ch("dst = cv2.cornerHarris(frame, 2, 3, 0.04)", f_view(s_harris_color), False,
                        "cornerHarris는 1채널(흑백) 영상만 받습니다. 컬러를 넣으면 에러가 나요.",
                        "cornerHarris needs a single-channel image; a color frame raises an error."),
                     ch("edges = cv2.Canny(gray, 100, 200)", f_view(s_canny_gray), False,
                        "엣지(선)를 뽑을 뿐, 코너 점을 찾는 함수가 아닙니다.",
                        "That finds edge lines, not corner points."),
                 ]),
        ]))

    ms.append(dict(
        id=4, name_ko="움직임 추적 + Kalman 예측", name_en="Track + Kalman predict", needs_hands=False,
        source=lambda raw: st_new(raw, raw),
        stages=[
            dict(short_ko="중심 좌표", short_en="Centroid", unit_name="Centroid",
                 ko="1단계: 피부색 물체(얼굴·손)의 중심 좌표를 구하세요. 추적의 첫 단계입니다.",
                 en="Step 1: get the center of the skin-colored object (face/hand). The first tracking step.",
                 setup=[("# cnt = 피부색 마스크에서 찾은 가장 큰 윤곽", "# cnt = largest contour of the skin mask"),
                        "# M = cv2.moments(cnt)"], metric=None,
                 choices=[
                     ch('cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]', make_centroid(_cen_ok), True,
                        "모멘트를 면적(m00)으로 나누면 무게중심입니다. 빨간 십자가 물체 한가운데에 찍혔어요.",
                        "Moments divided by the area (m00) give the centroid: the red cross sits at the object's center."),
                     ch('cx, cy = M["m01"] / M["m00"], M["m10"] / M["m00"]', make_centroid(_cen_swap), False,
                        "x와 y를 바꿔서 점이 대각선 반대편에 찍힙니다.",
                        "x and y are swapped, so the point lands mirrored across the diagonal."),
                     ch('cx, cy = M["m10"], M["m01"]', make_centroid(_cen_raw), False,
                        "면적으로 나누지 않아서 값이 엄청 커지고 화면 밖으로 나갑니다.",
                        "Without dividing by the area the values are huge and off screen."),
                     ch("x, y, w, h = cv2.boundingRect(cnt); cx, cy = x, y", make_centroid(_cen_corner), False,
                        "사각형의 왼쪽 위 모서리일 뿐 중심이 아닙니다.",
                        "That is the box's top-left corner, not the center."),
                     ch("cx, cy = cnt[0][0]", make_centroid(_cen_first), False,
                        "윤곽선의 첫 번째 점이라 테두리 위의 아무 점입니다.",
                        "That is the first point of the outline, somewhere on the border."),
                 ]),
            dict(short_ko="Kalman 모델", short_en="Model", unit_name="Kalman",
                 ko="2단계: 좌표를 Kalman 필터에 넣어 움직임을 예측하세요. 물체를 좌우로 움직여 보세요. 노란 화살표가 이동 방향을 앞서야 합니다.",
                 en="Step 2: feed the point to a Kalman filter. Move the object; the yellow arrow should lead its motion.",
                 setup=["# kf = cv2.KalmanFilter(4, 2)   # state [x, y, vx, vy], measurement [x, y]",
                        "# kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)   # H: measure x, y only"], metric=None,
                 choices=[
                     ch("kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)", make_kalman(dict(trans=KF_CV)), True,
                        "등속 모델: 다음 위치 = 현재 위치 + 속도. 그래서 노란 예측 화살표가 움직이는 방향을 앞서 가리킵니다.",
                        "Constant velocity: next position = position + velocity, so the yellow arrow leads the motion."),
                     ch("kf.transitionMatrix = np.eye(4, dtype=np.float32)", make_kalman(dict(trans=KF_ID)), False,
                        "속도가 위치에 더해지지 않아서 예측이 없습니다. 화살표가 거의 0이고 추정점은 뒤처져요.",
                        "Velocity never feeds position, so there is no prediction: the arrow is ~0 and the estimate lags."),
                     ch("kf = cv2.KalmanFilter(2, 2)   # state = [x, y]", make_kalman(dict(dim=2)), False,
                        "상태에 속도가 없어서 위치만 따라갑니다. 앞을 내다보는 예측 화살표가 만들어지지 않아요.",
                        "No velocity in the state: it only follows the position, with no look-ahead."),
                     ch("kf.transitionMatrix = np.array([[1,0,-1,0],[0,1,0,-1],[0,0,1,0],[0,0,0,1]], np.float32)", make_kalman(dict(trans=KF_NEG)), False,
                        "속도 부호가 반대라 예측 화살표가 이동 방향의 반대쪽을 가리킵니다.",
                        "The velocity sign is flipped, so the arrow points against the motion."),
                     ch("kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e5", make_kalman(dict(trans=KF_CV, r=1e5)), False,
                        "측정 노이즈를 엄청 크게 잡으면 측정을 거의 믿지 않아서 추정점이 제자리에 머뭅니다.",
                        "A huge measurement noise makes the filter ignore the data; the estimate stays put."),
                 ]),
            dict(short_ko="가려짐 대응", short_en="Occlusion", unit_name="Occlusion",
                 ko="3단계: 물체가 가려져 측정값이 사라지는 순간(화면이 어두워짐)에도 위치를 이어 가야 합니다. 올바른 처리는?",
                 en="Step 3: when the object is hidden (screen dims) there is no measurement. What is the right handling?",
                 setup=[("# 3초마다 1.2초 동안 측정값 z 가 사라집니다 (가려짐 흉내)",
                         "# every 3 s the measurement z vanishes for 1.2 s (occlusion)"),
                        ("# 측정이 있을 때는 predict() 후 correct(z) 를 합니다",
                         "# with a measurement: predict() then correct(z)")], metric=None,
                 choices=[
                     ch("est = kf.predict()   # no measurement: use the prediction",
                        make_kalman(dict(trans=KF_CV), "predict", True), True,
                        "측정이 없을 땐 correct 없이 predict 결과만 씁니다. 가려진 동안에도 초록 점이 진행 방향으로 계속 움직여요.",
                        "With no measurement use predict() alone: the green dot keeps moving along its path."),
                     ch("est = kf.correct(last_z)   # reuse the last measurement",
                        make_kalman(dict(trans=KF_CV), "hold", True), False,
                        "옛 측정값으로 계속 보정하면 추정점이 마지막 위치에 붙어서 멈춰 버립니다.",
                        "Correcting with the stale measurement pins the estimate at the last position."),
                     ch("est = (0, 0)   # no measurement: origin", make_kalman(dict(trans=KF_CV), "origin", True), False,
                        "측정이 없다고 0으로 두면 추정점이 화면 왼쪽 위 구석으로 튑니다.",
                        "Setting zero sends the estimate to the top-left corner."),
                     ch("kf = cv2.KalmanFilter(4, 2)   # rebuild the filter when lost",
                        make_kalman(dict(trans=KF_CV), "reset", True), False,
                        "필터를 새로 만들면 속도 정보가 사라지고 가려진 동안 추정 자체가 끊깁니다.",
                        "Rebuilding the filter loses the velocity and there is no estimate while hidden."),
                     ch("est = kf.correct(np.zeros((2, 1), np.float32))   # correct with zeros",
                        make_kalman(dict(trans=KF_CV), "zeros", True), False,
                        "0을 측정값으로 넣으면 추정점이 왼쪽 위 구석으로 끌려갑니다.",
                        "Feeding zeros as a measurement drags the estimate toward the top-left corner."),
                 ]),
        ]))

    ms.append(dict(
        # needs_hands=True: 메뉴에서 mediapipe 와 모델 파일을 확인해서, 없으면 이 미션 항목만 회색으로
        # 비활성이 된다.
        id=5, name_ko="MediaPipe 손 인식 파이프라인", name_en="MediaPipe hand pipeline", needs_hands=True,
        source=lambda raw: st_new(raw, raw),
        stages=[
            dict(short_ko="RGB 입력", short_en="RGB input", unit_name="RGB input",
                 ko="1단계: 카메라 프레임(BGR)을 MediaPipe 모델에 넣으려 합니다. 손을 보여 주고, 올바른 입력 준비는?",
                 en="Step 1: feed the camera frame (BGR) to MediaPipe. Show your hand. Which input prep is right?",
                 setup=["# image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)"], metric=None,
                 choices=[
                     ch("rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)", h_detect, True,
                        "MediaPipe의 SRGB는 R, G, B 순서를 기대합니다. 변환해서 넣었더니 손이 검출됐어요.",
                        "MediaPipe's SRGB expects R, G, B order. After converting, the hand is detected."),
                     ch("rgb = frame   # keep BGR as it is", h_model_view(lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2RGB),
                                                                    "what the model sees: R and B swapped"), False,
                        "BGR을 그대로 넣으면 모델은 R과 B가 뒤바뀐 화면을 보게 됩니다. 피부가 파랗게 보이는 그 화면이에요.",
                        "Passing BGR as-is makes the model see R and B swapped (blue skin)."),
                     ch("rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)", h_model_view(s_bgr2gray,
                                                                                    "1 channel: not RGB"), False,
                        "흑백은 1채널입니다. 3채널 RGB 영상이 아니라서 SRGB 입력으로 맞지 않아요.",
                        "Gray is 1 channel, not the 3-channel RGB image SRGB expects."),
                     ch("rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)", h_model_view(s_bgr2hsv,
                                                                                   "HSV values, not RGB"), False,
                        "HSV 값은 RGB가 아닙니다. 모델 입력으로 쓰면 색이 완전히 틀어져요.",
                        "HSV values are not RGB; the colors would be completely wrong."),
                     ch("rgb = cv2.resize(frame, (64, 48))", h_model_view(
                         lambda f: cv2.resize(cv2.resize(f, (64, 48)), (VID_W, VID_H), interpolation=cv2.INTER_NEAREST),
                         "64x48: the hand is tiny"), False,
                        "해상도를 너무 줄이면 손이 몇 픽셀로 뭉개져서 검출이 어려워집니다.",
                        "Shrinking to 64x48 leaves the hand a few pixels; detection gets hard."),
                 ]),
            dict(short_ko="픽셀 좌표", short_en="Pixels", unit_name="Pixels",
                 ko="2단계: 랜드마크 좌표는 0~1로 정규화돼 있습니다. 화면 픽셀 좌표로 바꾸는 올바른 코드는?",
                 en="Step 2: landmarks are normalized to 0-1. Which code converts them to pixel coordinates?",
                 setup=["# h, w = frame.shape[:2]",
                        ("# lm : 손 하나의 랜드마크 21개 (p.x, p.y 는 0~1)", "# lm : 21 landmarks of one hand (p.x, p.y in 0-1)")], metric=None,
                 choices=[
                     ch("pts = [(int(p.x * w), int(p.y * h)) for p in lm]", make_pix(_pix_ok), True,
                        "x에는 폭(w), y에는 높이(h)를 곱해야 합니다. 뼈대가 손 위에 정확히 얹혔어요.",
                        "Multiply x by the width and y by the height: the skeleton sits on the hand."),
                     ch("pts = [(int(p.x), int(p.y)) for p in lm]", make_pix(_pix_noscale), False,
                        "0~1 값을 그대로 정수로 바꾸면 전부 0이 되어 점이 왼쪽 위 한 곳에 몰립니다.",
                        "Casting 0-1 values to int gives 0, so every point piles up at the top-left."),
                     ch("pts = [(int(p.x * h), int(p.y * w)) for p in lm]", make_pix(_pix_hw), False,
                        "폭과 높이를 바꿔 곱해서 뼈대가 찌그러지고 엉뚱한 곳에 그려집니다.",
                        "Width and height are swapped, so the skeleton is squashed and misplaced."),
                     ch("pts = [(int(p.y * w), int(p.x * h)) for p in lm]", make_pix(_pix_swap), False,
                        "x와 y를 바꿔서 뼈대가 대각선으로 뒤집혀 그려집니다.",
                        "x and y are swapped, so the skeleton is mirrored across the diagonal."),
                     ch("pts = [(int(p.x * w), int((1 - p.y) * h)) for p in lm]", make_pix(_pix_flip), False,
                        "y를 1에서 빼면 위아래가 뒤집힙니다. 이미지 좌표는 위쪽이 0이에요.",
                        "1 - y flips it vertically. Image coordinates start at 0 on top."),
                 ]),
            dict(short_ko="손가락 판정", short_en="Fingers", unit_name="Fingers",
                 ko="3단계: 관절 각도로 손가락이 펴졌는지 판정합니다. 손을 폈다 쥐어 보세요. 올바른 조건은?",
                 en="Step 3: decide if a finger is extended from its joint angle. Open and close your hand.",
                 setup=[("# angle = 손가락 관절 세 점의 각도(도)", "# angle = joint angle of three finger landmarks (deg)"),
                        "# angle = calculate_angle(lm[a], lm[b], lm[c])"], metric=None,
                 choices=[
                     ch("extended = angle >= 160", make_rule(lambda a: a >= ANGLE_THRESHOLD), True,
                        "쭉 펴진 손가락은 관절 각도가 180도에 가깝습니다. 160도 이상이면 폈다고 보고 세요.",
                        "A straight finger has a joint angle near 180. Count it as extended at 160 or more."),
                     ch("extended = angle <= 160", make_rule(lambda a: a <= ANGLE_THRESHOLD), False,
                        "부등호가 반대라 구부린 손가락을 센 것입니다. 손을 펴면 0, 쥐면 5가 나와요.",
                        "The inequality is reversed: it counts folded fingers (open hand = 0, fist = 5)."),
                     ch("extended = angle >= 20", make_rule(lambda a: a >= 20), False,
                        "기준이 너무 낮아서 구부린 손가락도 전부 펴진 것으로 셉니다. 항상 5가 나와요.",
                        "The bar is too low; even folded fingers count, so it always says 5."),
                     ch("extended = angle >= 179", make_rule(lambda a: a >= 179), False,
                        "기준이 너무 엄격해서 활짝 편 손가락도 거의 인정이 안 됩니다. 늘 0에 가까워요.",
                        "The bar is too strict; even an open hand barely counts, so it says ~0."),
                 ]),
        ]))

    # "완성 코드" 만들기. 미션을 끝내면 정답 보기를 이어 붙인 코드가 화면과 터미널에 나온다.
    # 정답 코드만 이으면 gray, hsv, kernel 같은 변수가 정의되지 않으므로, 단계 앞에 붙일 준비 줄을 (미션
    # 번호, 단계 번호) -> 줄들 로 적어 둔다. 이 줄들은 보여 주기용 텍스트일 뿐 실행되지 않는다.
    final_pre = {
        (1, 2): ["gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)"],
        (2, 1): ["hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)"],
        (2, 2): ["kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))"],
        (2, 3): ["contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)"],
        (3, 0): ["clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))",
                 "L, a, b = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))"],
        (3, 1): ["sharp_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], np.float32)"],
        (3, 2): ["gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)",
                 "dst = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)",
                 "dst = cv2.dilate(dst, None)"],
        (4, 0): ["M = cv2.moments(cnt)   # cnt = largest skin contour"],
        (4, 1): ["kf = cv2.KalmanFilter(4, 2)",
                 "kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)"],
        (4, 2): ["# when no measurement z arrives this frame:"],
        (5, 1): ["h, w = frame.shape[:2]"],
        (5, 2): ["angle = calculate_angle(lm[a], lm[b], lm[c])"],
    }

    # 각 단계의 정답 보기 함수를 stg["_correct_fn"] 으로 저장해 둔다. 이것이 핵심이다:
    # Quiz._stage_input() 이 "앞 단계들의 정답 함수"를 이 이름으로 꺼내 매 프레임 다시 적용한다.
    for m in ms:
        lines = []
        for k, stg in enumerate(m["stages"]):
            right = [c for c in stg["choices"] if c[2]][0]
            stg["_correct_fn"] = right[1]
            lines.extend(final_pre.get((m["id"], k), []))
            lines.append(right[0])
        m["final_lines"] = lines
    return ms


# [패널 좌표]  아래 값들은 "패널(오른쪽 460x480) 안"의 좌표다. 캔버스 좌표로 바꾸려면 x 에 VID_W(640)를
# 더한다. GEO_* = (첫 보기 버튼의 y, 버튼 높이, 버튼 간격). 기초 문제와 미션은 문제 문장/setup 줄이
# 차지하는 높이가 달라서 배치가 다르다. 버튼 i 의 위치는 btn_rect(i, geo) 가 계산하고, 그리기와 클릭
# 판정이 같은 함수를 쓰므로 둘이 어긋나지 않는다.
BTN_X, BTN_W = 12, PANEL_W - 24
CODE_X = 46
CODE_W = BTN_W - CODE_X - 8
CODE_SCALES = (0.43, 0.40)
SETUP_SCALES = (0.36, 0.34)
CLEAR_SCALES = (0.40, 0.38, 0.36, 0.34)
GEO_BASIC = (172, 48, 4)
GEO_MISSION = (204, 41, 4)
NEXT_RECT = (PANEL_W - 140, 440, PANEL_W - 12, 472)
SUM_RESTART = (12, 410, 148, 446)
SUM_WRONG = (156, 410, 302, 446)
SUM_MENU = (310, 410, 448, 446)
MENU_Y0, MENU_H, MENU_GAP = 44, 44, 4
MENU_KINDS = ["basic", "m1", "m2", "m3", "m4", "m5", "all", "free"]


def btn_rect(i, geo=GEO_BASIC):
    y0, h, gap = geo
    y = y0 + i * (h + gap)
    return (BTN_X, y, BTN_X + BTN_W, y + h)


def menu_rect(i):
    y = MENU_Y0 + i * (MENU_H + MENU_GAP)
    return (BTN_X, y, BTN_X + BTN_W, y + MENU_H)


def in_rect(r, x, y):
    return r[0] <= x < r[2] and r[1] <= y < r[3]


def _setup_text(line, ko):
    return line if isinstance(line, str) else (line[0] if ko else line[1])


# 퀴즈 전체의 상태 머신. state 는 4가지:
#   "menu"     세트를 고르는 메뉴                     -> start() 로 "ask"
#   "ask"      문제가 나와 있고 보기를 고르는 중       -> 정답이면 "solved", 오답이면 "ask" 유지
#                                                       (틀린 보기는 빨갛게 잠김)
#   "solved"   정답을 맞힌 상태 (0.5초 페이드 + 해설)  -> SPACE / NEXT 로 다음 문제의 "ask"
#   "summary"  모든 문제를 끝낸 결과 화면             -> R 처음부터 / W 틀린 문제만 / M 메뉴
# 입력(key / click / move)은 상태를 바꾸고, 그리기(render)는 상태를 읽기만 한다.
class Quiz:
    def __init__(self, text, seed=None, hands=None, start="menu"):
        self.text = text
        self.rng = random.Random(seed)
        self.bank = build_questions()
        self.missions = build_missions()
        self.hands = hands

        # 미션 5 의 h_detect 가 쓸 수 있도록 HandTracker 를 공유 슬롯에 등록한다. (FreePlay 와 같은
        # 객체를 쓰므로 모델은 한 번만 로드)
        _HANDS["tracker"] = hands

        # 캐시 3종: 문제 패널의 정적 부분, 해설 배너, 메뉴. 글씨 렌더링(특히 한글)이 비싸서 한 번 그린
        # 것은 기억해 두고 재사용한다.
        self.base_cache = {}
        self.banner_cache = {}
        self.menu_cache = {}
        self.sum_cache = None
        self.hover = None
        self.kind = "all"
        self.menu_msg = ""
        self._avail = None
        self.questions = []
        self.geo = GEO_BASIC
        self._reset_run()
        self.state = "menu"
        if start != "menu":
            self.start(start)

    # 한 번의 도전(세트)을 시작할 때 초기화하는 값들. pctx / cctx 는 단계 함수들의 ctx 저장소다:
    #   pctx  앞 단계 "정답 함수"들의 ctx  (키: (미션 id, 단계 번호))
    #   cctx  지금 고른 보기의 ctx         (키: (문제 id, 보기 번호))
    #   ema   측정 숫자를 부드럽게 만드는 평균값 저장소
    def _reset_run(self):
        self.qi = 0
        self.score = 0
        self.streak = 0
        self.results = {}
        self.picked = None
        self.wrong = set()
        self.tries = 0
        self.t_solved = None
        self.pctx = {}
        self.cctx = {}
        self.ema = {}
        self.cleared = set()

    # 손 인식을 쓸 수 있는지 한 번만 확인해서 기억한다 (mediapipe 설치 여부 + 모델 파일 존재). mediapipe
    # 자체는 import 하지 않는다.
    def hands_ok(self):
        if self._avail is None:
            if self.hands is None:
                self._avail = (False, "no hand tracker")
            else:
                self._avail = self.hands.available()
        return self._avail

    def mission_ok(self, m):
        return (not m["needs_hands"]) or self.hands_ok()[0]

    def _selected_missions(self, kind):
        if kind in ("missions", "all"):
            return list(self.missions)
        return [m for m in self.missions if "m%d" % m["id"] == kind]

    # 선택한 세트를 "문제 목록"으로 펼친다. 기초 문제 8개는 그대로, 미션은 단계 하나하나가 문제 하나가
    # 된다. 미션 단계 문제는 dict(stg) 를 복사한 뒤 mission(소속 미션), stage_i(몇 번째 단계), id = 100
    # x 미션번호 + 단계 로 표시한다. 손 인식이 불가능하면 needs_hands 미션은 목록에서 제외한다.
    def _build_list(self, kind):
        qs = []
        if kind in ("basic", "all"):
            for i, q in enumerate(self.bank):
                d = dict(q)
                d.update(id=i, group="basic", mission=None, stage_i=0)
                qs.append(d)
        for m in self._selected_missions(kind):
            if not self.mission_ok(m):
                continue
            for k, stg in enumerate(m["stages"]):
                d = dict(stg)
                d.update(id=100 * m["id"] + k, group="m%d" % m["id"], mission=m, stage_i=k,
                         unit="M%d" % m["id"], unit_name=m["name_en"], problem=None)
                qs.append(d)
        return qs

    # 메뉴에서 세트를 고른 순간. "free" 면 문자열을 돌려줘서 App 이 자유 실습 모드로 바꾸게 한다.
    # 미션 5 를 고르면 이때 처음으로 hands.ensure() 가 mediapipe 를 import 하고 모델을 로드한다 (잠깐
    # 멈출 수 있음). 실패하면: 단일 미션이면 이유를 메뉴에 표시하고 돌아가고, 전체 도전이면 손 미션만
    # 빼고 진행한다.
    def start(self, kind):
        if kind == "free":
            return "free"
        sel = self._selected_missions(kind)
        if kind.startswith("m") and kind != "missions" and sel and not self.mission_ok(sel[0]):
            self.menu_msg = "Unavailable: " + self.hands_ok()[1]
            return None
        if any(m["needs_hands"] and self.mission_ok(m) for m in sel):
            if not self.hands.ensure():
                self._avail = (False, self.hands.err)
                if kind.startswith("m") and kind != "missions":
                    self.menu_msg = "MediaPipe unavailable: " + self.hands.err
                    return None
        self.kind = kind
        self.menu_msg = ""
        self.reset_session()
        return None

    # 문제 목록을 만들고 각 문제의 보기 순서를 섞는다 (정답 위치가 매번 바뀜). subset 은 "틀린 문제만
    # 다시" 용 id 목록.
    def reset_session(self, subset=None):
        qs = self._build_list(self.kind)
        if subset is not None:
            keep = set(subset)
            qs = [q for q in qs if q["id"] in keep]
        for q in qs:
            ch_ = list(q["choices"])
            self.rng.shuffle(ch_)
            q["choices"] = ch_
        self.questions = qs
        self._reset_run()
        self.sum_cache = None
        if qs:
            self.state = "ask"
            self._enter_question()
        else:
            self.state = "summary"

    # 새 문제로 넘어갈 때마다 선택 / 오답 표시 / 시도 횟수 / ctx 를 모두 초기화하고, 문제
    # 종류(미션/기초)에 맞는 버튼 배치를 고른다.
    def _enter_question(self):
        self.picked = None
        self.wrong = set()
        self.tries = 0
        self.t_solved = None
        self.hover = None
        self.pctx = {}
        self.cctx = {}
        self.ema = {}
        self.geo = GEO_MISSION if self.current()["mission"] else GEO_BASIC

    def current(self):
        return self.questions[self.qi]

    def is_last_stage(self, q):
        return q["mission"] is not None and q["stage_i"] == len(q["mission"]["stages"]) - 1

    # 보기를 골랐을 때의 처리. 이미 틀린 보기이거나 "ask" 상태가 아니면 무시한다.
    # 점수: 첫 시도에 맞히면 10점, 두 번째 5점, 그 이후 0점. streak 은 "첫 시도 정답" 연속 횟수
    # (틀리거나 건너뛰면 0). 정답이면 "solved" 로 바꾸고 t_solved 에 시각을 기록한다 (페이드 시작 시점).
    # 미션의 마지막 단계면 완성 코드를 출력한다. cctx / ema 를 비우는 것은 보기를 바꿀 때마다 Kalman
    # 같은 상태 있는 처리를 처음부터 다시 시작하기 위해서다.
    def pick(self, idx, now):
        if self.state != "ask" or idx is None:
            return
        q = self.current()
        if not (0 <= idx < len(q["choices"])) or idx in self.wrong:
            return
        ok = q["choices"][idx][2]
        self.tries += 1
        self.picked = idx
        self.cctx.pop((q["id"], idx), None)
        self.ema.pop("n", None)
        if ok:
            self.state = "solved"
            self.t_solved = now
            pts = 10 if self.tries == 1 else (5 if self.tries == 2 else 0)
            self.score += pts
            self.streak = self.streak + 1 if self.tries == 1 else 0
            self.results[q["id"]] = dict(tries=self.tries, pts=pts, skipped=False)
            if self.is_last_stage(q):
                self._announce_clear(q["mission"])
        else:
            self.wrong.add(idx)
            self.streak = 0

    # 미션 클리어 시 터미널에 완성 코드를 한 번만 출력한다 (cleared 집합으로 중복 방지).
    def _announce_clear(self, m):
        if m["id"] in self.cleared:
            return
        self.cleared.add(m["id"])
        print("\n[MISSION %d CLEAR] %s -- final pipeline:" % (m["id"], m["name_en"]))
        for ln in m["final_lines"]:
            print("    " + ln)

    def skip(self):
        if self.state != "ask":
            return
        q = self.current()
        self.results[q["id"]] = dict(tries=self.tries, pts=0, skipped=True)
        self.streak = 0
        self._advance()

    def next(self):
        if self.state == "solved":
            self._advance()

    def _advance(self):
        self.qi += 1
        if self.qi >= len(self.questions):
            self.state = "summary"
            self.sum_cache = None
        else:
            self.state = "ask"
            self._enter_question()

    def restart_all(self):
        self.reset_session()

    def restart_wrong(self):
        ids = [q["id"] for q in self.questions
               if self.results.get(q["id"], {}).get("pts", 0) < 10]
        if ids:
            self.reset_session(ids)

    def to_menu(self):
        self.state = "menu"
        self.hover = None
        self.menu_msg = ""

    # 키 입력은 상태별로 나눠 처리한다. chr(k).lower() 로 대소문자를 구분하지 않는다. 메뉴에서 1~8 은
    # 세트 선택, 문제 중에는 1~9 가 보기 선택(pick), SPACE(32) / ENTER(13) 는 다음 문제, S 는 건너뛰기,
    # M 은 메뉴.
    def key(self, k, now):
        c = chr(k).lower() if 0 <= k < 128 else ""
        if self.state == "menu":
            if "1" <= c <= "8":
                return self.start(MENU_KINDS[int(c) - 1])
            return None
        if c == "m":
            self.to_menu()
            return None
        if self.state == "summary":
            if c == "r":
                self.restart_all()
            elif c == "w":
                self.restart_wrong()
            return None
        if "1" <= c <= "9":
            self.pick(int(c) - 1, now)
        elif k in (32, 13):
            self.next()
        elif c == "s":
            self.skip()
        return None

    # 마우스 클릭. 캔버스 x 가 영상 영역(< VID_W)이면 무시하고, 패널 영역이면 px = x - VID_W 로 패널
    # 좌표로 바꿔서 in_rect() 로 어떤 버튼인지 찾는다. 이 x, y 는 이미 배율이 나눠진 논리 좌표다
    # (App.mouse_cb 참고).
    def click(self, x, y, now):
        if x < VID_W:
            return None
        px, py = x - VID_W, y
        if self.state == "menu":
            for i in range(len(MENU_KINDS)):
                if in_rect(menu_rect(i), px, py):
                    return self.start(MENU_KINDS[i])
            return None
        if self.state == "summary":
            if in_rect(SUM_RESTART, px, py):
                self.restart_all()
            elif in_rect(SUM_WRONG, px, py):
                self.restart_wrong()
            elif in_rect(SUM_MENU, px, py):
                self.to_menu()
            return None
        if self.state == "solved" and in_rect(NEXT_RECT, px, py):
            self.next()
            return None
        q = self.current()
        for i in range(len(q["choices"])):
            if in_rect(btn_rect(i, self.geo), px, py):
                self.pick(i, now)
                return None
        return None

    # 마우스 이동: 지금 어느 버튼 위에 있는지 hover 에 기록한다 (그릴 때 버튼 색을 밝게 하는 용도).
    def move(self, x, y):
        self.hover = None
        if x < VID_W or self.state == "summary":
            return
        px, py = x - VID_W, y
        if self.state == "menu":
            for i in range(len(MENU_KINDS)):
                if in_rect(menu_rect(i), px, py):
                    self.hover = i
                    return
            return
        q = self.current()
        for i in range(len(q["choices"])):
            if in_rect(btn_rect(i, self.geo), px, py):
                self.hover = i
                return
        if in_rect(NEXT_RECT, px, py):
            self.hover = "next"

    # 상태에 맞는 1100x480 논리 캔버스 한 장을 만든다. 메뉴는 라이브 영상 + 메뉴 패널, 결과는 어둡게 한
    # 영상 + 결과 패널, 문제 중에는 _video() 가 만든 영상 + _panel() 이 만든 패널이다.
    def render(self, raw, now):
        canvas = np.zeros((WIN_H, WIN_W, 3), np.uint8)
        if self.state == "menu":
            canvas[:, :VID_W] = raw
            outlined(canvas, "LIVE CAMERA - pick a set on the right", (10, 24), 0.6, C_GOOD)
            canvas[:, VID_W:] = self._menu_panel()
            return canvas
        if self.state == "summary":
            canvas[:, :VID_W] = (raw.astype(np.float32) * 0.45).astype(np.uint8)
            canvas[:, VID_W:] = self._summary_panel()
            return canvas
        q = self.current()
        view, tag, tag_col = self._video(q, raw, now)
        canvas[:, :VID_W] = view
        canvas[:, VID_W:] = self._panel(q)
        outlined(canvas, tag, (10, 24), 0.6, tag_col)
        return canvas

    # 기초 문제 보기 결과를 영상으로 올리는 방법 (kind). "mask" 면 마스크를 base 에 씌우고, 아니면
    # to_view() 로 통일.
    def _post(self, base, out, kind):
        if kind == "mask":
            return apply_mask(base, out)
        return to_view(out)

    # [미션의 핵심]  지금 단계의 "입력 영상"을 만든다.
    #   1) m["source"](raw)                      카메라 원본으로 망가진 시작 상태를 만들고
    #   2) 앞 단계 j = 0 .. stage_i-1 의 정답 함수 _correct_fn 을 차례로 적용한다
    # 결과를 저장해 두지 않고 "매 프레임 처음부터 다시 계산"하는 이유는 입력이 살아 있는 카메라 영상이기
    # 때문이다. 각 앞 단계는 자기 ctx(pctx)를 계속 유지해서 Kalman 같은 상태가 프레임 사이에 이어진다.
    def _stage_input(self, q, raw, now):
        m = q["mission"]
        st = m["source"](raw)
        for j in range(q["stage_i"]):
            ctx = self.pctx.setdefault((m["id"], j), {})
            ctx["now"] = now
            st = m["stages"][j]["_correct_fn"](st, ctx)
        return st

    # 지수 이동 평균: 새 값 25% + 이전 평균 75%. 측정 숫자가 프레임마다 깜빡이지 않게 부드럽게 만든다.
    def _ema(self, key, v):
        old = self.ema.get(key)
        self.ema[key] = v if old is None else 0.75 * old + 0.25 * v
        return self.ema[key]

    # 왼쪽 영상 한 장을 만든다. 돌려주는 값 = (영상, 왼쪽 위 태그 글씨, 태그 색).
    #   base  "문제 화면":  미션이면 _stage_input() 결과, 기초면 problem(raw) 로 망가뜨린 영상
    #   보기를 고르기 전   base 만 보여 준다.
    #   보기를 고른 후     그 보기를 base 에 실제로 적용한 결과(res)를 보여 준다.
    # 보기 실행 중 예외가 나면(예: Canny 에 float32) 프로그램이 죽는 대신 base 를 보여 주고 에러 문구를
    # 영상에 표시한다.
    def _video(self, q, raw, now):
        err = None
        broken = None
        st_in = None
        if q["mission"] is not None:
            try:
                st_in = self._stage_input(q, raw, now)
                base = vis_img(st_in)
            except Exception as e:
                base = to_view(raw)
                err = short_err(e)
        else:
            broken = q["problem"](raw) if q["problem"] else raw
            base = to_view(broken)
        if q["mission"] is not None:
            head = "STEP %d/%d" % (q["stage_i"] + 1, len(q["mission"]["stages"]))
        else:
            head = "PROBLEM VIEW"
        if self.picked is None:
            view = base.copy()
            self._metric(view, q, base, None)
            if err:
                outlined(view, clip_text("ERROR: " + err, VID_W - 20, 0.45), (10, VID_H - 12), 0.45, C_BAD, 1)
            return view, head + " - pick the fix", C_BAD

        # 고른 보기를 실행한다. kind == "st" (미션)는 fn(st_in, ctx) 에 입력 상태와 보기별 ctx 를
        # 넘기고, 그 외(기초)는 fn(망가진 프레임) 결과를 _post() 로 영상에 올린다.
        code, fn, ok, why_ko, why_en, kind = q["choices"][self.picked]
        try:
            if kind == "st":
                if st_in is None:
                    raise RuntimeError(err or "previous step failed")
                ctx = self.cctx.setdefault((q["id"], self.picked), {})
                ctx["now"] = now
                res = vis_img(fn(st_in, ctx))
            else:
                res = self._post(base, fn(broken), kind)
            err = None
        except Exception as e:
            res = base
            err = short_err(e)

        # 정답이면 base -> res 로 0.5초(FADE_SEC) 동안 cv2.addWeighted 로 서서히 섞는다 (t: 0 -> 1).
        # 오답이면 res 를 그대로 보여 준다.
        if ok and self.state == "solved":
            t = min(1.0, (now - self.t_solved) / FADE_SEC)
            view = cv2.addWeighted(base, 1.0 - t, res, t, 0)
            tag, col = ("MISSION CLEAR" if self.is_last_stage(q) else "SOLVED"), C_GOOD
        else:
            view = res.copy()
            tag, col = "YOUR PICK - wrong", C_WARN

        # 오른쪽 위에 "BEFORE" 작은 화면: 고른 보기를 적용하기 전의 문제 화면을 함께 보여 줘서 전후
        # 비교가 되게 한다.
        ins = cv2.resize(base, (200, 150), interpolation=cv2.INTER_AREA)
        view[10:160, VID_W - 210:VID_W - 10] = ins
        cv2.rectangle(view, (VID_W - 210, 10), (VID_W - 10, 160), (255, 255, 255), 1)
        outlined(view, "BEFORE", (VID_W - 204, 28), 0.5, C_TEXT, 1)
        self._metric(view, q, base, res, ok)
        self._banner(view, q, ok, err)
        return view, tag, col

    # 선명도 / 노이즈 / 대비 숫자를 영상 왼쪽 위에 표시. "before -> after" 형태이고, 정답이면서 실제로
    # 좋아졌을 때만 초록, 그 외에는 주황으로 쓴다. 값은 _ema 로 부드럽게 만든다.
    def _metric(self, view, q, base, res, ok=False):
        key = q.get("metric")
        if not key:
            return
        name, fn, higher = METRICS[key]
        vb = self._ema("b", fn(base))
        label = "%s  %.0f" % (name, vb) if key != "noise" else "%s  %.1f" % (name, vb)
        outlined(view, label, (10, 70), 0.55, C_TEXT, 1)
        if res is None:
            return
        vn = self._ema("n", fn(res))
        better = vn > vb if higher else vn < vb
        w = cv2.getTextSize(label, FONT, 0.55, 1)[0][0]
        if vn > 9999:
            txt = " -> >9999"
        else:
            txt = " -> %.0f" % vn if key != "noise" else " -> %.1f" % vn
        outlined(view, txt, (10 + w, 70), 0.55, C_GOOD if (ok and better) else C_WARN, 2)

    # 영상 아래쪽에 정답 / 오답 해설을 깔아 준다. 해설 글씨는 (문제 id, 고른 보기, 언어)별로 캐시한다.
    # 배경은 원본 30% + 색(정답 초록 / 오답 빨강) 70% 로 섞어서 글씨가 읽히게 한다.
    def _banner(self, view, q, ok, err):
        key = (q["id"], self.picked, self.text.ko)
        if key not in self.banner_cache:
            ch_ = q["choices"][self.picked]
            why = ch_[3] if self.text.ko else ch_[4]
            last = self.is_last_stage(q)
            if self.text.ko:
                head = ("미션 완료!  " if last else "정답!  ") if ok else "오답  "
            else:
                head = ("MISSION CLEAR!  " if last else "Correct!  ") if ok else "Wrong.  "
            cov, h = self.text.block(head + why, VID_W - 28, 17, max_lines=5)
            self.banner_cache[key] = (cov, h + 18 + (16 if ok else 0))
        cov, bh = self.banner_cache[key]
        y0 = VID_H - bh
        tint = np.array((30, 95, 30) if ok else (30, 30, 120), np.float32)
        roi = view[y0:VID_H]
        roi[:] = (roi.astype(np.float32) * 0.30 + tint * 0.70).astype(np.uint8)
        paste_cov(view, cov, 14, y0 + 9, (255, 255, 255))
        if err:
            outlined(view, clip_text("ERROR: " + err, VID_W - 20, 0.45), (10, y0 - 10), 0.45, C_BAD, 1)
        if ok and self.state == "solved":
            put(view, "SPACE / click NEXT", (VID_W - 170, VID_H - 6), 0.45, (255, 255, 255), 1)

    # 문제 문장 아래의 setup 코드 줄들을 그린다. ASCII 줄은 putText(빠름, fit_lines 로 최대 2줄),
    # 한글이 섞인 줄은 TextRenderer 로 그린다. 첫 보기 버튼(limit)과 겹치기 전에 멈춘다.
    def _setup_block(self, p, lines, top, limit):
        for line in lines:
            s = _setup_text(line, self.text.ko)
            if top > limit - 12:
                break
            if s.isascii():
                subs, sc = fit_lines(s, PANEL_W - 28, SETUP_SCALES, "#   " if s.startswith("#") else "  ")
                for sub in subs:
                    put(p, sub, (14, top + 11), sc, C_ACCENT)
                    top += 13
            elif self.text.ko:
                top = self.text.draw(p, s, 14, top, PANEL_W - 28, 12, C_ACCENT, max_lines=2)
        return top

    # 패널에서 "문제가 바뀔 때만 변하는 정적 부분"(미션 단계 칩, 문제 문장, setup 줄)을 그려서 캐시한다.
    # 캐시 키 = (문제 id, 언어). 단계 칩: 끝난 단계 초록 / 현재 단계 파랑+테두리 / 남은 단계 회색.
    def _panel_base(self, q):
        key = (q["id"], self.text.ko)
        if key in self.base_cache:
            return self.base_cache[key]
        p = np.full((WIN_H, PANEL_W, 3), C_PANEL, np.uint8)
        qtext = q["ko"] if self.text.ko else q["en"]
        if q["mission"] is not None:
            m = q["mission"]
            n = len(m["stages"])
            gap = 6
            cw = (PANEL_W - 24 - gap * (n - 1)) // n
            for k, stg in enumerate(m["stages"]):
                x0 = 12 + k * (cw + gap)
                y0, y1 = 36, 58
                done, cur = k < q["stage_i"], k == q["stage_i"]
                fill = C_BTN_RIGHT if done else ((95, 72, 22) if cur else (46, 44, 42))
                cv2.rectangle(p, (x0, y0), (x0 + cw, y1), fill, -1)
                cv2.rectangle(p, (x0, y0), (x0 + cw, y1), C_ACCENT if cur else (90, 90, 90), 1)
                label = "%d %s" % (k + 1, stg["short_ko"] if self.text.ko else stg["short_en"])
                self.text.draw(p, label, x0 + 5, y0 + 2, cw - 8, 13, C_TEXT, max_lines=1)
            yq = self.text.draw(p, qtext, 14, 64, PANEL_W - 28, 16, C_TEXT, max_lines=4)
            self._setup_block(p, q["setup"], yq + 4, GEO_MISSION[0])
        else:
            yq = self.text.draw(p, qtext, 14, 40, PANEL_W - 28, 17, C_TEXT, max_lines=4)
            self._setup_block(p, q["setup"], max(121, yq + 2), GEO_BASIC[0])
        self.base_cache[key] = p
        return p

    # 패널 한 프레임 = 캐시된 정적 패널의 복사본 + 변하는 요소(제목줄, 보기 버튼, 점수, NEXT).
    # .copy() 가 꼭 필요하다: 캐시 원본 위에 직접 그리면 다음 프레임에 이전 버튼 색이 남는다.
    # 버튼 색 우선순위: 정답으로 고른 것(초록) > 이미 틀린 것(빨강) > 마우스가 올라간 것(밝은 회색) >
    # 기본.
    def _panel(self, q):
        p = self._panel_base(q).copy()
        cv2.rectangle(p, (0, 0), (PANEL_W, 30), C_HEADER, -1)
        put(p, "Q %d/%d" % (self.qi + 1, len(self.questions)), (12, 21), 0.6, C_WARN, 2)
        put(p, "%s | %s" % (q["unit"], q["unit_name"][:26]), (110, 21), 0.5, C_TEXT)
        n = len(q["choices"])
        if self.state == "solved" and self.is_last_stage(q):
            self._clear_box(p, q["mission"])
        else:
            for i, (code, fn, ok, wk, we, kind) in enumerate(q["choices"]):
                x0, y0, x1, y1 = btn_rect(i, self.geo)
                if self.state == "solved" and i == self.picked:
                    col = C_BTN_RIGHT
                elif i in self.wrong:
                    col = C_BTN_WRONG
                elif self.hover == i and self.state == "ask":
                    col = C_BTN_HOVER
                else:
                    col = C_BTN
                cv2.rectangle(p, (x0, y0), (x1, y1), col, -1)
                cv2.rectangle(p, (x0, y0), (x1, y1), (110, 110, 110), 1)
                cv2.circle(p, (x0 + 22, (y0 + y1) // 2), 14, (20, 20, 20), -1, cv2.LINE_AA)
                put(p, str(i + 1), (x0 + 16, (y0 + y1) // 2 + 6), 0.6, C_WARN, 2)
                lines, sc = fit_lines(code, CODE_W, CODE_SCALES, "  ")
                ty = (y0 + y1) // 2 - (len(lines) - 1) * 9 + 5
                for ln in lines:
                    put(p, ln, (x0 + CODE_X, ty), sc, C_TEXT)
                    ty += 18
        put(p, "SCORE %d   STREAK %d" % (self.score, self.streak), (12, 452), 0.55, C_GOOD, 1)
        if self.state == "solved":
            x0, y0, x1, y1 = NEXT_RECT
            col = C_BTN_HOVER if self.hover == "next" else C_BTN_RIGHT
            cv2.rectangle(p, (x0, y0), (x1, y1), col, -1)
            put(p, "NEXT >>", (x0 + 22, y0 + 22), 0.6, C_TEXT, 2)
        else:
            put(p, "click a choice or press 1-%d" % n, (12, 464), 0.4, C_DIM)
            put(p, "S skip | M menu | P free | +/- size | Q quit", (12, 476), 0.4, C_DIM)
        return p

    # 미션을 끝냈을 때 보기 버튼 자리에 "완성 코드" 박스를 그린다. 줄이 많으면 글자 크기를 CLEAR_SCALES
    # 에서 한 단계씩 줄여 가며 박스 높이에 들어가는 첫 크기를 고른다.
    def _clear_box(self, p, m):
        y0 = GEO_MISSION[0] - 4
        cv2.rectangle(p, (BTN_X, y0), (BTN_X + BTN_W, 434), (40, 58, 40), -1)
        cv2.rectangle(p, (BTN_X, y0), (BTN_X + BTN_W, 434), C_GOOD, 1)
        put(p, "MISSION CLEAR - your pipeline", (BTN_X + 10, y0 + 20), 0.5, C_GOOD, 1)
        first, last = y0 + 40, 428
        for sc in CLEAR_SCALES:
            pitch = int(sc * 37)
            rows = [fit_lines(ln, BTN_W - 20, (sc,), "  ")[0] for ln in m["final_lines"]]
            n_rows = sum(len(r) for r in rows)
            if (n_rows - 1) * pitch + 3 * (len(rows) - 1) <= last - first:
                break
        pitch = max(pitch, min(18, (last - first - 3 * (len(rows) - 1)) // max(n_rows - 1, 1)))
        y = first
        for r in rows:
            for sub in r:
                put(p, sub, (BTN_X + 10, y), sc, C_TEXT)
                y += pitch
            y += 3

    def _menu_labels(self):
        ko = self.text.ko
        items = []
        items.append(("기초 퀴즈  ·  8문제" if ko else "Basics: 8 questions",
                      "A Adjust  B Color  C Filter  D Edge  E HSV  F Morph  G CLAHE", True, ""))
        for m in self.missions:
            ok = self.mission_ok(m)
            name = m["name_ko"] if ko else m["name_en"]
            title = ("미션 %d  %s" if ko else "Mission %d  %s") % (m["id"], name)
            sub = " > ".join(s["short_en"] for s in m["stages"])
            items.append((title, sub, ok, "" if ok else self.hands_ok()[1]))
        items.append(("전체 도전  ·  기초 + 미션 모두" if ko else "Everything: basics + all missions",
                      "%d questions in total" % (len(self.bank) + sum(
                          len(m["stages"]) for m in self.missions if self.mission_ok(m))), True, ""))
        items.append(("자유 실습 모드  (키로 효과 켜기)" if ko else "Free play (toggle effects with keys)",
                      "flip blur sharpen edges CLAHE contour Harris Kalman hand", True, ""))
        return items

    # 메뉴 패널의 정적 부분을 캐시한다. 키에 "손 인식 가능 여부"를 넣은 이유: 미션 5 항목의
    # 활성/비활성이 나중에 바뀔 수 있어서.
    def _menu_base(self):
        key = (self.text.ko, self._avail is None or self.hands_ok()[0])
        if key in self.menu_cache:
            return self.menu_cache[key]
        p = np.full((WIN_H, PANEL_W, 3), C_PANEL, np.uint8)
        cv2.rectangle(p, (0, 0), (PANEL_W, 30), C_HEADER, -1)
        put(p, "VISION QUIZ LAB", (12, 21), 0.6, C_WARN, 2)
        put(p, "choose a set", (300, 21), 0.45, C_DIM)
        for i, (title, sub, ok, why) in enumerate(self._menu_labels()):
            x0, y0, x1, y1 = menu_rect(i)
            cv2.rectangle(p, (x0, y0), (x1, y1), C_BTN if ok else (38, 38, 38), -1)
            cv2.rectangle(p, (x0, y0), (x1, y1), (110, 110, 110) if ok else (70, 70, 70), 1)
            cv2.circle(p, (x0 + 20, (y0 + y1) // 2), 13, (20, 20, 20), -1, cv2.LINE_AA)
            put(p, str(i + 1), (x0 + 15, (y0 + y1) // 2 + 6), 0.55, C_WARN if ok else C_DIM, 2)
            self.text.draw(p, title, x0 + 42, y0 + 4, BTN_W - 52, 14, C_TEXT if ok else C_DIM, max_lines=1)
            put(p, (sub if ok else "needs: " + why)[:60], (x0 + 42, y0 + 36), 0.34, C_DIM if ok else C_BAD)
        put(p, "press 1-8 or click   |   P free   |   +/- size   |   Q quit", (12, 472), 0.4, C_DIM)
        self.menu_cache[key] = p
        return p

    def _menu_panel(self):
        p = self._menu_base().copy()
        if isinstance(self.hover, int) and 0 <= self.hover < len(MENU_KINDS) and self.state == "menu":
            x0, y0, x1, y1 = menu_rect(self.hover)
            cv2.rectangle(p, (x0, y0), (x1, y1), C_ACCENT, 2)
        if self.menu_msg:
            y = 438
            lines, sc = fit_lines(ascii_text(self.menu_msg), PANEL_W - 24, (0.4, 0.36), "  ")
            for ln in lines:
                put(p, ln, (12, y), sc, C_BAD)
                y += 14
        return p

    # 결과 화면. 기초 문제는 한 줄씩(첫 시도 +10 / 2번째 +5 / +0 / 건너뜀), 미션은 단계별 점 + 합계로
    # 요약한다. 한 번 그리면 캐시한다.
    def _summary_panel(self):
        if self.sum_cache is not None:
            return self.sum_cache
        p = np.full((WIN_H, PANEL_W, 3), C_PANEL, np.uint8)
        cv2.rectangle(p, (0, 0), (PANEL_W, 30), C_HEADER, -1)
        put(p, "RESULT", (12, 21), 0.6, C_WARN, 2)
        total = 10 * len(self.questions)
        put(p, "SCORE %d / %d" % (self.score, total), (12, 62), 0.9, C_GOOD, 2)
        y = 92
        seen = set()
        for q in self.questions:
            if q["mission"] is None:
                r = self.results.get(q["id"], dict(tries=0, pts=0, skipped=True))
                if r["skipped"]:
                    label, col = "SKIPPED", C_BAD
                elif r["tries"] == 1:
                    label, col = "FIRST TRY +10", C_GOOD
                elif r["pts"] > 0:
                    label, col = "2ND TRY +5", C_WARN
                else:
                    label, col = "SOLVED +0", C_WARN
                put(p, "%d  %s | %s" % (q["id"] + 1, q["unit"], q["unit_name"]), (14, y), 0.5, C_TEXT)
                put(p, label, (300, y), 0.5, col, 1)
                y += 22
            else:
                m = q["mission"]
                if m["id"] in seen:
                    continue
                seen.add(m["id"])
                stages = [x for x in self.questions if x["mission"] is m]
                put(p, "M%d  %s" % (m["id"], m["name_en"][:26]), (14, y), 0.5, C_TEXT)
                pts = 0
                for k, sq in enumerate(stages):
                    r = self.results.get(sq["id"], dict(tries=0, pts=0, skipped=True))
                    pts += r["pts"]
                    col = C_BAD if r["skipped"] else (C_GOOD if r["tries"] == 1 else C_WARN)
                    cv2.circle(p, (262 + k * 18, y - 5), 6, col, -1, cv2.LINE_AA)
                put(p, "+%d/%d" % (pts, 10 * len(stages)), (362, y), 0.5, C_GOOD if pts == 10 * len(stages) else C_WARN, 1)
                y += 22
        n_bad = sum(1 for q in self.questions if self.results.get(q["id"], {}).get("pts", 0) < 10)
        if n_bad == 0 and self.questions:
            put(p, "PERFECT!", (14, 396), 0.7, C_GOOD, 2)
        for rect, label, on in ((SUM_RESTART, "[R] Restart", True),
                                (SUM_WRONG, "[W] Wrong (%d)" % n_bad, n_bad > 0),
                                (SUM_MENU, "[M] Menu", True)):
            x0, y0, x1, y1 = rect
            cv2.rectangle(p, (x0, y0), (x1, y1), C_BTN if on else (40, 40, 40), -1)
            cv2.rectangle(p, (x0, y0), (x1, y1), (110, 110, 110), 1)
            put(p, label, (x0 + 10, y0 + 24), 0.5, C_TEXT if on else C_DIM)
        put(p, "P free mode | +/- size | Q quit", (12, 472), 0.42, C_DIM)
        self.sum_cache = p
        return p


# 세 점 p1-p2-p3 의 각도 (p2 가 꼭짓점, 단위 도). p2 에서 p1, p3 로 향하는 벡터 v1, v2 의 내적으로
# 구한다:
#   cos(theta) = (v1 . v2) / (|v1| |v2|),   theta = arccos(...)
# z(깊이)까지 포함한 3차원 각도다. 아주 작은 오차로 cos 이 1 을 넘으면 arccos 가 NaN 이 되므로
# np.clip(-1, 1) 로 막는다.
def calculate_angle(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if m1 == 0 or m2 == 0:
        return 0.0
    cos = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


# 펴진 손가락 개수 (FreePlay 의 "Fingers: n"). 노트북의 기준(각도 >= 160)을 그대로 쓴다.
# 엄지는 관절 구조가 달라서 이 기준이 정확하지 않을 수 있다.
def count_extended_fingers(landmarks):
    n = 0
    for a, b, c in FINGER_ANGLE_POINTS:
        if calculate_angle(landmarks[a], landmarks[b], landmarks[c]) >= ANGLE_THRESHOLD:
            n += 1
    return n


# MediaPipe HandLandmarker 를 감싼 클래스. 설계 포인트 두 가지:
#   1) mediapipe 를 지연 로드한다. 파일 맨 위에서 import 하지 않고 ensure() 안에서 import 하므로,
#      mediapipe 가 없어도 프로그램은 뜨고 손 기능만 "사용 불가"가 된다.
#   2) tried 플래그: 한 번 시도해서 실패하면 다시 시도하지 않는다 (매 프레임 실패 로그를 반복하지
#     않으려고).
class HandTracker:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.det = None
        self.ok = False
        self.err = ""
        self.tried = False
        self.t0 = time.monotonic()
        self.last_ts = -1

    # 모델 파일 후보를 앞에서부터 확인해 처음 존재하는 경로를 쓴다:
    # --hand-model 값 -> 현재 폴더 기준 -> 이 파일 폴더 기준 -> 이 파일의 한 단계 위(레포 루트) 기준 ->
    # 이 파일 옆의 hand_landmarker.task. (~/vision-llm 같은 다른 폴더는 찾지 않는다. 필요하면
    # --hand-model 로 지정)
    def _find_model(self):
        here = os.path.dirname(os.path.abspath(__file__))
        cands = []
        if self.model_path:
            cands.append(self.model_path)
        cands += [HAND_MODEL,
                  os.path.join(here, HAND_MODEL),
                  os.path.join(here, "..", HAND_MODEL),
                  os.path.join(here, "hand_landmarker.task")]
        for c in cands:
            if os.path.isfile(c):
                return c
        return None

    # "쓸 수 있는가?"만 가볍게 확인한다. importlib.util.find_spec 은 mediapipe 를 실제로 import 하지
    # 않고 설치 여부만 보므로 빠르다. 메뉴를 그릴 때(hands_ok) 불리므로, 여기서 무거운 import 를 하면
    # 시작이 느려진다.
    def available(self):
        if self.ok:
            return True, ""
        if self.tried:
            return False, self.err
        try:
            import importlib.util
            if importlib.util.find_spec("mediapipe") is None:
                return False, "mediapipe not installed (pip install mediapipe)"
        except Exception:
            pass
        if self._find_model() is None:
            return False, "hand_landmarker.task not found (src/models/MediaPipe/)"
        return True, ""

    # 처음 필요할 때 단 한 번 mediapipe 를 import 하고 HandLandmarker 를 만든다.
    #   RunningMode.VIDEO   연속된 프레임을 추적하는 모드. detect_for_video 에 "증가하는
    #     타임스탬프(ms)"가 필요하다
    #   num_hands=2, 세 신뢰도 임계값 0.5
    # 실패하면 err 에 이유를 ASCII 로 저장한다 (화면 글씨가 putText 라서).
    def ensure(self):
        if self.tried:
            return self.ok
        self.tried = True
        try:
            import mediapipe as mp
            path = self._find_model()
            if path is None:
                self.err = "hand_landmarker.task not found (src/models/MediaPipe/)"
                return False
            opts = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=path),
                running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
                min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5)
            self.det = mp.tasks.vision.HandLandmarker.create_from_options(opts)
            self.mp = mp
            self.ok = True
        except Exception as e:
            self.err = ascii_text("%s: %s" % (type(e).__name__, str(e)[:80]))
        return self.ok

    # 프레임 한 장에서 손 랜드마크를 구한다 (손 하나 = 랜드마크 21개, 좌표 x, y 는 0~1로 정규화).
    #   BGR -> RGB 변환 (MediaPipe 의 SRGB 형식은 R, G, B 순서를 기대한다)
    #   np.ascontiguousarray: 메모리가 연속인 배열이어야 mp.Image 가 받아 준다
    #   타임스탬프 ts = max(직전 + 1, 시작 후 경과 ms): 같은 ms 안에 두 번 불려도 항상 증가하도록
    #     보장한다
    def detect(self, frame_bgr):
        if not self.ensure():
            return []
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        ts = max(self.last_ts + 1, int((time.monotonic() - self.t0) * 1000))
        self.last_ts = ts
        return self.det.detect_for_video(image, ts).hand_landmarks

    def close(self):
        if self.det is not None:
            self.det.close()


# 색 분할 프리셋 (HSV 범위). "red" 는 빨강의 H 가 0 근처와 179 근처 양쪽에 걸쳐 있어서 범위 2개를 OR 로
# 합친다.
SEG_PRESETS = [
    ("off", None),
    ("skin", [((0, 40, 70), (20, 255, 255))]),
    ("green", [((40, 60, 50), (85, 255, 255))]),
    ("orange", [((8, 100, 100), (25, 255, 255))]),
    ("red (2 ranges)", [((0, 100, 70), (8, 255, 255)), ((172, 100, 70), (179, 255, 255))]),
    ("blue", [((95, 80, 50), (130, 255, 255))]),
]
FLIP_NAMES = ["off", "LR  flip(.,1)", "UD  flip(.,0)", "BOTH flip(.,-1)"]
FLIP_CODES = [None, 1, 0, -1]
SHARP_NAMES = ["off", "kernel 5 (4-neighbor)", "kernel 9 (8-neighbor)", "kernel 3 (weak)"]
SHARP_KS = [None, SHARP_K, SHARP_K8, SHARP_KW]
EDGE_NAMES = ["off", "Sobel", "Canny"]
HSV_NAMES = ["off", "H", "S", "V"]

# [ ] 키로 조절하는 값들: v 현재값, lo / hi 범위, step 한 번에 바뀌는 양. (블러 커널 크기는 홀수여야
# 해서 step=2)
PARAMS = {
    "box": dict(v=11, lo=1, hi=51, step=2),
    "gauss": dict(v=11, lo=1, hi=51, step=2),
    "clip": dict(v=3.0, lo=1.0, hi=10.0, step=0.5),
    "canny": dict(v=100, lo=10, hi=250, step=10),
    "ahead": dict(v=8, lo=1, hi=30, step=1),
}


# 퀴즈 없이 키로 효과를 켜고 끄며 실험하는 모드. 켜진 효과마다 해당 OpenCV 코드 한 줄이 self.codes 에
# 쌓여서 패널의 "CODE RUNNING" 에 표시된다. sel 은 [ ] 로 값을 바꿀 효과(가장 최근에 켠 것)다.
class FreePlay:
    def __init__(self, hand_model=None):
        self.hands = HandTracker(hand_model)
        self.reset()

    def reset(self):
        self.flip = 0
        self.gray = False
        self.hsvch = 0
        self.box = False
        self.gauss = False
        self.sharp = 0
        self.clahe = False
        self.edge = 0
        self.seg = 0
        self.morph = False
        self.contour = False
        self.harris = False
        self.kalman = False
        self.kt = "color"
        self.hand = False
        self.pip = False
        self.p = {k: dict(v) for k, v in PARAMS.items()}
        self.sel = None
        self.kf = None
        self.kf_lost = 0
        self.trail = deque(maxlen=40)
        self.msg = ""
        self.msg_until = 0.0
        self.codes = []
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def say(self, text):
        self.msg = text
        self.msg_until = time.monotonic() + 3.0

    def key(self, k):
        c = chr(k).lower() if 0 <= k < 128 else ""
        if c == "1":
            self.flip = (self.flip + 1) % 4
        elif c == "2":
            self.gray = not self.gray
        elif c == "3":
            self.hsvch = (self.hsvch + 1) % 4
        elif c == "4":
            self.box = not self.box
            self.sel = "box" if self.box else self.sel
        elif c == "5":
            self.gauss = not self.gauss
            self.sel = "gauss" if self.gauss else self.sel
        elif c == "6":
            self.sharp = (self.sharp + 1) % 4
        elif c == "7":
            self.edge = (self.edge + 1) % 3
            if self.edge == 2:
                self.sel = "canny"
        elif c == "8":
            self.clahe = not self.clahe
            self.sel = "clip" if self.clahe else self.sel
        elif c == "9":
            self.seg = (self.seg + 1) % len(SEG_PRESETS)
        elif c == "m":
            self.morph = not self.morph
        elif c == "c":
            self.contour = not self.contour
        elif c == "h":
            self.harris = not self.harris
        elif c == "k":
            self.kalman = not self.kalman
            self.kf = None
            self.trail.clear()
            self.sel = "ahead" if self.kalman else self.sel
        elif c == "t":
            if self.kt == "color":
                if self.hands.ensure():
                    self.kt = "finger"
                    self.say("Kalman target: index fingertip")
                else:
                    self.say("MediaPipe unavailable: " + self.hands.err)
            else:
                self.kt = "color"
                self.say("Kalman target: color blob")
            self.kf = None
            self.trail.clear()
        elif c == "g":
            self.hand = not self.hand
            if self.hand and not self.hands.ensure():
                self.say("MediaPipe unavailable: " + self.hands.err)
                self.hand = False
        elif c == "v":
            self.pip = not self.pip
        elif c == "r":
            self.reset()
        elif c in ("[", "]") and self.sel:
            sp = self.p[self.sel]
            sp["v"] = min(sp["hi"], max(sp["lo"], sp["v"] + (sp["step"] if c == "]" else -sp["step"])))

    # HSV 범위에 드는 픽셀의 마스크. 범위가 여러 개면 OR 로 합친다. morph 가 켜져 있으면 OPEN(점 제거)
    # 뒤 CLOSE(구멍 메우기).
    def seg_mask(self, img_bgr):
        name, ranges = SEG_PRESETS[self.seg] if self.seg else SEG_PRESETS[1]
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = None
        for lo, hi in ranges:
            m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        if self.morph:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    @staticmethod
    def largest_contour(mask, min_area=400):
        res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = res[0] if len(res) == 2 else res[1]
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        return c if cv2.contourArea(c) >= min_area else None

    # 미션 4 의 Kalman 과 같은 등속도 모델. meas = (x, y) 측정 또는 None. 돌려주는 값 = (x, y, vx, vy)
    # 또는 None. 첫 측정에서 필터를 만들고, 매 프레임 predict -> (측정이 있으면 correct). 측정이 없으면
    # predict 결과를 추정으로 쓰고, 25프레임 넘게 못 보면 추적을 끝낸다.
    def _kalman_step(self, meas):
        if self.kf is None:
            if meas is None:
                return None
            kf = cv2.KalmanFilter(4, 2)
            kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
            kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
            kf.processNoiseCov = np.diag([1e-2, 1e-2, 5e-2, 5e-2]).astype(np.float32)
            kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4.0
            kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0
            kf.statePost = np.array([[meas[0]], [meas[1]], [0], [0]], np.float32)
            kf.statePre = kf.statePost.copy()
            self.kf = kf
            self.kf_lost = 0
            self.trail.clear()
        pred = self.kf.predict()
        if meas is not None:
            est = self.kf.correct(np.array([[meas[0]], [meas[1]]], np.float32))
            self.kf_lost = 0
        else:
            est = pred
            self.kf_lost += 1
            if self.kf_lost > 25:
                self.kf = None
                self.trail.clear()
                return None
        return float(est[0, 0]), float(est[1, 0]), float(est[2, 0]), float(est[3, 0])

    # 켜 둔 효과를 정해진 순서로 적용한다:
    #   반전 -> 흑백 -> HSV 채널 -> box 블러 -> Gaussian 블러 -> 샤프닝 -> CLAHE -> 엣지 -> (색 분할) ->
    #     윤곽 -> Harris -> 손 -> Kalman -> PIP
    # 주의: 색 분할 / 윤곽 / Harris / 손 인식은 "반전만 된 원본(base)"으로 계산한다. 블러나 엣지가 이미
    # 적용된 img 로 검출하면 결과가 망가지기 때문이다. 그림은 효과가 적용된 img 위에 그린다.
    def process(self, raw):
        self.codes = []

        # base: 반전만 적용한 영상 (검출 전용). img: 효과를 누적해서 적용하는 작업 영상.
        base = raw if self.flip == 0 else cv2.flip(raw, FLIP_CODES[self.flip])
        if self.flip:
            self.codes.append("frame = cv2.flip(frame, %d)" % FLIP_CODES[self.flip])
        img = base.copy()
        h, w = img.shape[:2]
        if self.gray:
            img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            self.codes.append("gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)")

        # H 채널은 0~179 범위라 그대로 보면 어둡다. 255/179 를 곱해서 0~255 로 늘려 보여 준다.
        if self.hsvch:
            ch = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))[self.hsvch - 1]
            if self.hsvch == 1:
                ch = (ch.astype(np.float32) * (255.0 / 179.0)).astype(np.uint8)
            img = cv2.cvtColor(ch, cv2.COLOR_GRAY2BGR)
            self.codes.append("h, s, v = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))   # show %s" % HSV_NAMES[self.hsvch].lower())
        if self.box:
            k = int(self.p["box"]["v"])
            img = cv2.blur(img, (k, k))
            self.codes.append("frame = cv2.blur(frame, (%d, %d))" % (k, k))
        if self.gauss:
            k = int(self.p["gauss"]["v"])
            img = cv2.GaussianBlur(img, (k, k), 0)
            self.codes.append("frame = cv2.GaussianBlur(frame, (%d, %d), 0)" % (k, k))
        if self.sharp:
            img = cv2.filter2D(img, -1, SHARP_KS[self.sharp])
            self.codes.append("frame = cv2.filter2D(frame, -1, sharp_kernel)   # %s" % SHARP_NAMES[self.sharp])
        if self.clahe:
            clip = float(self.p["clip"]["v"])
            self._clahe.setClipLimit(clip)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            img = cv2.cvtColor(cv2.merge([self._clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
            self.codes.append("clahe = cv2.createCLAHE(clipLimit=%.1f, tileGridSize=(8, 8))   # on L" % clip)

        # Sobel: x, y 방향 1차 미분(음수가 생기므로 CV_64F) -> magnitude 로 크기 -> /4 로 눈금을 맞춰
        # uint8 로. Canny: 낮은 임계값 lo 와 높은 임계값 lo x 2.
        if self.edge:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self.edge == 1:
                gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
                mag = np.clip(cv2.magnitude(gx, gy) / 4.0, 0, 255).astype(np.uint8)
                self.codes.append("mag = cv2.magnitude(gx, gy)   # gx, gy = Sobel x / y derivatives")
            else:
                lo = int(self.p["canny"]["v"])
                mag = cv2.Canny(g, lo, lo * 2)
                self.codes.append("edges = cv2.Canny(gray, %d, %d)" % (lo, lo * 2))
            img = cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)

        # 여기서부터는 검출 단계. 마스크를 base 에서 만들고, 색 분할이 켜져 있으면 img 에 씌운다.
        mask = None
        need_mask = self.seg or self.contour or (self.kalman and self.kt == "color")
        if need_mask:
            mask = self.seg_mask(base)
        if self.seg:
            img = cv2.bitwise_and(img, img, mask=mask)
            self.codes.append("mask = cv2.inRange(hsv, lower, upper)   # %s%s" % (
                SEG_PRESETS[self.seg][0], " + open/close" if self.morph else ""))
        blob = None
        if self.contour or (self.kalman and self.kt == "color"):
            cnt = self.largest_contour(mask)
            if cnt is not None:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    blob = (M["m10"] / M["m00"], M["m01"] / M["m00"])
                if self.contour:
                    cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
                    (cx, cy), r = cv2.minEnclosingCircle(cnt)
                    cv2.circle(img, (int(cx), int(cy)), int(r), (0, 200, 255), 2)
                    if blob:
                        cv2.circle(img, (int(blob[0]), int(blob[1])), 5, (0, 0, 255), -1)
            if self.contour:
                self.codes.append("cnt = max(contours, key=cv2.contourArea); M = cv2.moments(cnt)")
        if self.harris:
            g32 = np.float32(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY))
            dst = cv2.dilate(cv2.cornerHarris(g32, 2, 3, 0.04), None)
            if dst.max() > 0:
                img[dst > 0.01 * dst.max()] = (0, 0, 255)
            self.codes.append("dst = cv2.dilate(cv2.cornerHarris(np.float32(gray), 2, 3, 0.04), None)")

        # 손 인식: G 키(뼈대 + 손가락 수) 또는 Kalman 대상이 손가락일 때만 detect 를 호출한다 (CPU
        # 에서는 비용이 크다). G 의 그림: 뼈대 선과 점은 미션 5 의 draw_hand 와 같은 방식이고, 손목
        # 아래에 "Fingers: n" 을 쓴다.
        hand_lm = None
        if self.hand or (self.kalman and self.kt == "finger"):
            hand_lm = self.hands.detect(base)
        if self.hand and hand_lm:
            for lm in hand_lm:
                pts = [(int(p.x * w), int(p.y * h)) for p in lm]
                for a, b in HAND_CONNECTIONS:
                    cv2.line(img, pts[a], pts[b], (0, 255, 0), 2)
                for i, pt in enumerate(pts):
                    cv2.circle(img, pt, 5 if i in (4, 8, 12, 16, 20) else 3,
                               (0, 0, 255) if i in (4, 8, 12, 16, 20) else (255, 0, 0), -1)
                n = count_extended_fingers(lm)
                outlined(img, "Fingers: %d" % n, (pts[0][0] - 30, pts[0][1] + 30), 0.7, C_WARN)
            self.codes.append("extended = calculate_angle(lm[a], lm[b], lm[c]) >= %d" % ANGLE_THRESHOLD)
        elif self.hand:
            self.codes.append("# no hand detected")

        # Kalman 측정값: kt == "finger" 이면 검지 끝(landmark 8)의 픽셀 좌표, 아니면 색 덩어리의
        # 중심(blob).
        if self.kalman:
            meas = None
            if self.kt == "finger":
                if hand_lm:
                    tip = hand_lm[0][8]
                    meas = (tip.x * w, tip.y * h)
            else:
                meas = blob
            est = self._kalman_step(meas)
            if meas is not None:
                cv2.circle(img, (int(meas[0]), int(meas[1])), 7, (0, 0, 255), 2)
            if est is not None:
                ex, ey, vx, vy = est
                self.trail.append((int(ex), int(ey)))
                if len(self.trail) > 1:
                    cv2.polylines(img, [np.array(self.trail, np.int32)], False, (255, 255, 0), 2)
                cv2.circle(img, (int(ex), int(ey)), 6, (0, 255, 0), -1)
                k = int(self.p["ahead"]["v"])
                fx, fy = int(ex + vx * k), int(ey + vy * k)
                cv2.arrowedLine(img, (int(ex), int(ey)), (fx, fy), (0, 255, 255), 3, tipLength=0.25)
                lost = "   (target lost: predicting)" if meas is None else ""
                outlined(img, "Kalman%s" % lost, (10, h - 12), 0.55, C_WARN, 1)
            self.codes.append("est = kf.predict(); kf.correct(z)   # z=%s, arrow=%d frames ahead" % (
                self.kt, int(self.p["ahead"]["v"])))

        if self.pip:
            small = cv2.resize(raw, (200, 150), interpolation=cv2.INTER_AREA)
            img[h - 160:h - 10, w - 210:w - 10] = small
            cv2.rectangle(img, (w - 210, h - 160), (w - 10, h - 10), (255, 255, 255), 1)
        return img

    # 자유 실습 한 프레임: process() 결과 영상 + 오른쪽 패널을 붙여 1100x480 캔버스를 만든다. 안내
    # 메시지(msg)는 3초 동안 표시.
    def render(self, raw, now):
        img = self.process(raw)
        canvas = np.zeros((WIN_H, WIN_W, 3), np.uint8)
        canvas[:, :VID_W] = img
        canvas[:, VID_W:] = self._panel()
        outlined(canvas, "FREE PLAY", (10, 24), 0.6, C_ACCENT)
        if self.msg and time.monotonic() < self.msg_until:
            outlined(canvas, self.msg[:70], (10, VID_H - 34), 0.45, C_WARN, 1)
        return canvas

    def _panel(self):
        p = np.full((WIN_H, PANEL_W, 3), C_PANEL, np.uint8)
        cv2.rectangle(p, (0, 0), (PANEL_W, 30), C_HEADER, -1)
        put(p, "FREE PLAY", (12, 21), 0.6, C_ACCENT, 2)
        put(p, "P: quiz | R: reset | Q: quit", (170, 21), 0.42, C_DIM)
        rows = [
            ("1", "flip (A)", FLIP_NAMES[self.flip], self.flip != 0),
            ("2", "gray (B)", "on" if self.gray else "off", self.gray),
            ("3", "hsv channel (B)", HSV_NAMES[self.hsvch], self.hsvch != 0),
            ("4", "box blur (C)", "k=%d" % self.p["box"]["v"] if self.box else "off", self.box),
            ("5", "gaussian blur (C)", "k=%d" % self.p["gauss"]["v"] if self.gauss else "off", self.gauss),
            ("6", "sharpen (C)", SHARP_NAMES[self.sharp], self.sharp != 0),
            ("7", "edges (D)", EDGE_NAMES[self.edge] + (" lo=%d" % self.p["canny"]["v"] if self.edge == 2 else ""), self.edge != 0),
            ("8", "CLAHE on L (G)", "clip=%.1f" % self.p["clip"]["v"] if self.clahe else "off", self.clahe),
            ("9", "color segment (E)", SEG_PRESETS[self.seg][0], self.seg != 0),
            ("M", "morph open+close (F)", "on" if self.morph else "off", self.morph),
            ("C", "contour (H)", "on" if self.contour else "off", self.contour),
            ("H", "harris (I)", "on" if self.harris else "off", self.harris),
            ("K", "kalman (L)", ("%d frames ahead" % self.p["ahead"]["v"]) if self.kalman else "off", self.kalman),
            ("T", "kalman target", self.kt, False),
            ("G", "hand + fingers (M)", "on" if self.hand else "off", self.hand),
            ("V", "original PIP", "on" if self.pip else "off", self.pip),
        ]
        y = 52
        for key, name, val, on in rows:
            col = C_GOOD if on else C_DIM
            put(p, "[%s]" % key, (12, y), 0.5, C_WARN)
            put(p, name, (52, y), 0.5, C_TEXT if on else C_DIM)
            put(p, val, (250, y), 0.5, col)
            y += 19
        adj = "[ ] adjust: %s = %s" % (self.sel, self.p[self.sel]["v"]) if self.sel else "[ ] adjust: toggle an effect first"
        put(p, adj, (12, y + 6), 0.45, C_ACCENT)
        y += 28
        put(p, "CODE RUNNING:", (12, y), 0.45, C_WARN)
        y += 14
        for line in self.codes[-6:]:
            put(p, line[:72], (12, y), 0.33, C_TEXT)
            y += 13
        return p


# 화면 해상도 읽기: xrandr --current (없으면 xdpyinfo) 출력을 정규식으로 해석한다. tkinter 없이도
# 동작하는 방법이다.
def _screen_size_x11():
    import subprocess
    for cmd in (["xrandr", "--current"], ["xdpyinfo"]):
        try:
            out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=3).stdout
            out = out.decode("utf-8", "replace")
        except Exception:
            continue
        for pat in (r"\bconnected primary (\d+)x(\d+)\+", r"\bconnected (\d+)x(\d+)\+",
                    r"current\s+(\d+)\s*x\s*(\d+)", r"dimensions:\s+(\d+)x(\d+)"):
            m = re.search(pat, out)
            if m:
                return int(m.group(1)), int(m.group(2))
    return None


def screen_size():
    try:
        if platform.system() == "Windows":
            import ctypes
            u = ctypes.windll.user32
            return int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1))
        sz = _screen_size_x11()
        if sz:
            return sz
        import tkinter
        r = tkinter.Tk()
        r.withdraw()
        w, h = r.winfo_screenwidth(), r.winfo_screenheight()
        r.destroy()
        return int(w), int(h)
    except Exception:
        return None


# 자동 창 배율:  min(화면 폭 x 0.96 / 1100,  (화면 높이 - 130) / 480)  을 1.0 ~ 2.2 로 제한하고 0.05
# 단위로 반올림. 130px 은 제목 줄 / 작업 표시줄 여유. 해상도를 못 읽으면 1.0.  (1920x1080 화면이면
# x1.70)
def auto_scale():
    sz = screen_size()
    if not sz:
        return 1.0
    sw, sh = sz
    s = min(sw * 0.96 / WIN_W, (sh - 130) / WIN_H)
    return max(1.0, min(2.2, round(s * 20) / 20.0))


# 창 하나를 관리한다: Quiz, FreePlay 두 모드를 갖고 있고 mode 로 어느 쪽을 그릴지 정한다.
# 두 모드가 같은 HandTracker(self.free.hands)를 공유해서 MediaPipe 모델을 한 번만 로드한다.
class App:
    def __init__(self, args):
        self.text = TextRenderer(args.lang)
        self.free = FreePlay(args.hand_model)
        self.quiz = Quiz(self.text, args.seed, hands=self.free.hands, start=getattr(args, "set", "menu"))
        self.mode = args.mode
        sc = getattr(args, "scale", 0) or 0
        self.scale = float(sc) if sc > 0 else auto_scale()
        self.scale = max(0.8, min(3.0, self.scale))
        self.fps = 0.0
        self._t_prev = None

    # 현재 모드의 논리 캔버스(1100x480)를 만들고 FPS 를 덧그린다. FPS 는 순간값 10% + 이전 평균 90% 의
    # 이동 평균.
    def render(self, frame, now):
        if self._t_prev is not None:
            dt = max(now - self._t_prev, 1e-3)
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt
        self._t_prev = now
        canvas = self.quiz.render(frame, now) if self.mode == "quiz" else self.free.render(frame, now)
        outlined(canvas, "%.0f FPS" % self.fps, (10, 46), 0.45, C_TEXT, 1)
        return canvas

    # 화면에 내보내기 직전에 확대 배율을 곱하는 유일한 곳. 확대는 INTER_LINEAR, 축소는 INTER_AREA 가
    # 무난하다. 창 모드가 WINDOW_AUTOSIZE 라서 "창 크기 = 이 이미지 크기"가 된다.
    def present(self, canvas):
        s = self.scale
        if abs(s - 1.0) < 0.01:
            return canvas
        interp = cv2.INTER_LINEAR if s > 1.0 else cv2.INTER_AREA
        return cv2.resize(canvas, (int(round(WIN_W * s)), int(round(WIN_H * s))), interpolation=interp)

    def set_scale(self, s):
        self.scale = max(0.8, min(3.0, round(s * 20) / 20.0))
        print("[ui] window scale x%.2f" % self.scale)

    def _act(self, action):
        if action == "free":
            self.mode = "free"

    # 공통 키: Q / ESC 는 False 를 돌려줘서 main 루프를 끝낸다. P 는 모드 전환, + - 0 은 창 크기.
    # 나머지는 현재 모드에 넘긴다.
    def key(self, k, now):
        if k in (ord("q"), 27):
            return False
        if k == ord("p"):
            self.mode = "free" if self.mode == "quiz" else "quiz"
            return True
        if k in (ord("+"), ord("=")):
            self.set_scale(self.scale + 0.1)
            return True
        if k in (ord("-"), ord("_")):
            self.set_scale(self.scale - 0.1)
            return True
        if k == ord("0"):
            self.set_scale(auto_scale())
            return True
        if self.mode == "quiz":
            self._act(self.quiz.key(k, now))
        else:
            self.free.key(k)
        return True

    # OpenCV 마우스 콜백. 받은 x, y 는 "배율이 곱해진 창 좌표"이므로 scale 로 나눠 논리 좌표(1100x480
    # 기준)로 되돌린 뒤 Quiz 에 넘긴다. 이렇게 하면 창 크기를 바꿔도 버튼 판정 코드는 달라질 필요가
    # 없다. 자유 실습 모드에서는 마우스를 쓰지 않는다.
    def mouse_cb(self, event, x, y, flags, param):
        if self.mode != "quiz":
            return
        x, y = int(x / self.scale), int(y / self.scale)
        if event == cv2.EVENT_MOUSEMOVE:
            self.quiz.move(x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self._act(self.quiz.click(x, y, time.monotonic()))

    def close(self):
        self.free.hands.close()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="AirBall Vision Quiz Lab (06_03, Jetson version) - no options needed")
    ap.add_argument("--csi", action="store_true", help="Jetson CSI camera (nvarguscamerasrc) instead of the USB camera")
    ap.add_argument("--video", help="use a video file instead of a camera")
    ap.add_argument("--image", help="use a still image instead of a camera")
    ap.add_argument("--demo", action="store_true", help="synthetic video, no camera needed")
    ap.add_argument("--mode", choices=["quiz", "free"], default="quiz")
    ap.add_argument("--set", choices=["menu", "basic", "missions", "all", "m1", "m2", "m3", "m4", "m5"],
                    default="menu", help="skip the menu and start this question set")
    ap.add_argument("--scale", type=float, default=0.0,
                    help="window size multiplier (default: fit the screen automatically; +/- keys change it live)")
    ap.add_argument("--lang", choices=["auto", "ko", "en"], default="auto")
    ap.add_argument("--seed", type=int, default=None, help="shuffle seed for the choices")
    ap.add_argument("--hand-model", default=None, help="path to hand_landmarker.task")
    return ap.parse_args(argv)


# 리눅스에서 DISPLAY(또는 WAYLAND_DISPLAY)가 없으면 창을 못 띄운다. SSH 로 접속한 경우가 대표적이라
# 안내문을 출력한다.
def check_display():
    if platform.system() != "Linux":
        return True
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return True
    print("[error] 화면(DISPLAY)을 찾을 수 없어 창을 띄울 수 없습니다.")
    print("        Jetson 에 모니터가 연결돼 있다면 그 데스크톱의 터미널에서 실행하세요.")
    print("        SSH 로 접속한 상태라면:  export DISPLAY=:0   를 입력한 뒤 다시 실행하면 Jetson 모니터에 창이 뜹니다.")
    return False


# 프레임을 못 읽는 상황을 견디는 횟수. 실패할 때마다 waitKey(30) 으로 30ms 쉬므로 100회 = 약 3초 (read()
# 자체가 기다리는 시간은 제외). 시작 직후 카메라가 아직 준비 안 된 경우를 넘기고, 계속 안 들어오면
# 오류로 종료한다.
READ_FAIL_LIMIT = 100


# 흐름:  인자 해석 -> DISPLAY 확인 -> 입력(FrameSource) 열기 -> 창 만들기 -> 루프(읽기 -> 그리기 -> 보여
# 주기 -> 키 처리). 종료 코드 rc 는 0 이 정상, 1 이 오류다. 프로그램은 Q / ESC 를 누르거나 오류가 날
# 때만 끝난다.
def main(argv=None):
    args = parse_args(argv)
    if not check_display():
        return 1
    try:
        src = FrameSource(args)
    except RuntimeError as e:
        print("[error] %s" % e)
        return 1
    app = App(args)
    print("[ui] Korean font: %s" % (app.text.font_path or "not found -> English UI  (Korean: sudo apt install fonts-nanum)"))
    print("[ui] window scale x%.2f  (+ / - / 0 to change)" % app.scale)

    # WINDOW_AUTOSIZE: 창 크기가 보여 주는 이미지 크기를 따라간다 (배율은 present() 가 이미지에 곱한다).
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
    try:
        cv2.moveWindow(WIN_NAME, 10, 10)
    except Exception:
        pass
    cv2.setMouseCallback(WIN_NAME, app.mouse_cb)
    rc = 0
    fails = 0
    warned = False
    try:
        while True:
            # 프레임 읽기. 실패하면 fails 를 올리고 3초쯤 기다린다 (그 사이에도 Q / ESC 로 끝낼 수 있게
            # waitKey 를 부른다). 읽기에 성공하면 fails 를 0 으로 되돌린다.
            ok, frame = src.read()
            if not ok:
                fails += 1
                if not warned:
                    print("[camera] no frame yet - waiting up to about 3 s ...")
                    warned = True
                if fails > READ_FAIL_LIMIT:
                    print("[error] 카메라에서 프레임을 읽지 못해 종료합니다. USB 연결 / 다른 프로그램의 카메라 사용 여부를 확인하세요.")
                    rc = 1
                    break
                if (cv2.waitKey(30) & 0xFF) in (ord("q"), 27):
                    break
                continue
            fails = 0

            # 한 프레임: 논리 캔버스 만들기 -> present() 로 배율 적용 -> imshow. waitKey(1) 은 화면을
            # 갱신하고 키를 받는다. 키 값 255 는 "아무 키도 안 눌림"이라 무시하고, app.key() 가 False(Q
            # / ESC)를 돌려주면 루프를 빠져나간다.
            canvas = app.render(frame, time.monotonic())
            cv2.imshow(WIN_NAME, app.present(canvas))
            k = cv2.waitKey(1) & 0xFF
            if k != 255 and not app.key(k, time.monotonic()):
                break
    # 어떤 이유로 끝나든(정상, Ctrl+C, 예외) 손 인식 모델 닫기 / 카메라 해제 / 창 닫기를 보장한다.
    finally:
        app.close()
        src.release()
        cv2.destroyAllWindows()
    return rc


if __name__ == "__main__":
    sys.exit(main())

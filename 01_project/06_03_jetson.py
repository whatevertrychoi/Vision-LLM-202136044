#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_03_jetson.py  -  Vision Quiz Lab, Jetson 버전  (AirBall project / Computer Vision 섹션 학습용)

실시간 카메라 영상이 일부러 "망가진 상태"로 나오고, 같은 창의 오른쪽 패널에
5지선다 문제가 뜹니다. 보기를 마우스로 클릭(또는 키 1~5)하면 그 함수가
실시간 영상에 바로 적용됩니다.
  - 정답이면 화면이 맑아지고(0.5초 페이드) 해설이 뜹니다.
  - 오답이면 그 함수의 (이상한) 결과가 영상에 그대로 보이고, 다시 고를 수 있습니다.

시작하면 메뉴가 뜹니다 (1~8 키 또는 클릭)
  1 기초 퀴즈 8문제 (노트북 A~G단원: 반전, 색공간, 필터, 엣지, HSV, 모폴로지, CLAHE)
  2~6 단계 미션 (앞 단계의 정답 결과가 다음 단계의 입력이 되는 파이프라인 문제)
      미션1 뒤집힘+노이즈 -> 뒤집기 -> 노이즈 제거 -> 윤곽선(Canny)
      미션2 색 바로잡기 -> 피부색 HSV 마스크 -> MORPH_OPEN -> 윤곽/중심(contour, moments)
      미션3 어두운 화면 -> CLAHE -> 샤프닝 -> Harris 코너
      미션4 피부색 중심 좌표 -> Kalman 모델 -> 가려짐 대응 (움직임 예측)
      미션5 MediaPipe 손: RGB 입력 -> 정규화 좌표 -> 픽셀 -> 손가락 각도 판정 (모델 파일 필요)
  7 전체 도전 (기초 + 미션 전부)   8 자유 실습 모드
미션을 끝내면 정답 보기를 이어 붙인 "완성 코드"가 화면과 터미널에 나옵니다.

이 파일은 06_03.py 를 Jetson 에서 바로 돌리도록 고친 버전입니다. 퀴즈 내용과 화면은 06_03.py 와 같고,
달라진 점은 아래 4가지입니다.
  1) 카메라: Jetson USB 카메라용 GStreamer 파이프라인(/dev/video0, YUY2 640x480 30fps)을
             05.py / 06.py 와 똑같이 사용합니다. 옵션 없이 실행하면 바로 이 카메라가 열립니다.
  2) 경로:   손 모델 / 한글 폰트를 "이 파일이 있는 폴더(01_project)" 기준으로 찾습니다.
             레포 루트가 아니라 01_project 안에서 실행해도 됩니다.
  3) 창 크기: Jetson 화면 해상도를 xrandr 로 읽어서 자동 배율을 정합니다 (tkinter 가 없어도 동작).
  4) 안정성: 카메라 프레임이 잠깐 안 들어와도(시작 직후 등) 바로 종료하지 않고 3초쯤 기다립니다.
             모니터가 없는 SSH 터미널이면 DISPLAY 안내를, 카메라가 안 열리면 원인 후보를 알려 줍니다.

화면 크기: 화면 해상도에 맞춰 자동으로 키워서 엽니다.
           --scale 1.6 으로 직접 지정하거나, 실행 중 + / - 키로 조절, 0 키로 자동 크기.

실행 (Jetson 의 데스크톱 터미널에서)
  git pull
  cd ~/Documents/git/vision-LLM-<학번>/01_project     # 레포 폴더 이름은 본인 것
  python 06_03_jetson.py                      # 옵션 없이 바로 실행 (USB 카메라 /dev/video0)
  python 06_03_jetson.py --demo               # 카메라 없이 가짜 영상으로 실행 (화면 / 클릭 점검용)
  python 06_03_jetson.py --video a.mp4        # 영상 파일로 실행
  python 06_03_jetson.py --image a.png        # 이미지 한 장으로 실행
  python 06_03_jetson.py --csi                # Jetson CSI 카메라(nvarguscamerasrc) - 이 파이프라인은 검증 전
  python 06_03_jetson.py --set m2             # 메뉴 건너뛰고 미션2부터 (basic / missions / all / m1~m5)
  python 06_03_jetson.py --mode free          # 자유 실습 모드로 시작
  python 06_03_jetson.py --scale 1.6          # 창 크기 배율 지정
  python 06_03_jetson.py --lang en            # 영어 UI (한글 폰트가 없을 때 자동 적용)

필요 패키지: opencv (GStreamer 가 켜진 빌드. 05.py / 06.py 가 돌아가는 환경이면 OK), numpy,
             Pillow (한글 표시용, 없으면 영어 UI)
선택 패키지: mediapipe + hand_landmarker.task  (미션5 / 자유 실습의 손 인식 / Kalman 손가락 추적)
             모델 위치: <레포 루트>/src/models/MediaPipe/hand_landmarker.task  (05.py / 06.py 와 같은 곳)
한글 폰트: Jetson 에 없으면  sudo apt install fonts-nanum   (또는 이 파일 옆에 ko_font.ttf 를 두세요)
           폰트가 없어도 영어 UI 로 정상 동작합니다.

키
  [메뉴]  1~8 또는 클릭: 세트 선택
  [퀴즈]  1~5 또는 클릭: 선택 | SPACE/ENTER: 다음 | S: 건너뛰기 | M: 메뉴
          결과 화면: R 처음부터 | W 틀린 문제만 | M 메뉴
  [공통]  P: 퀴즈/자유 전환 | + - 0: 창 크기 | Q/ESC: 종료
  [자유]  1 반전  2 흑백  3 HSV채널  4 box블러  5 Gaussian블러  6 샤프닝
          7 엣지(Sobel/Canny)  8 CLAHE  9 색분할  M 모폴로지  C 윤곽
          H Harris  K Kalman  T Kalman대상  G 손인식  V 원본 PIP
          [ ] 파라미터 조절  R 초기화  P 퀴즈로  Q 종료
"""
######################################################################
# [이 파일을 읽는 방법]  -- 아래 주석은 모두 "설명용"이며 실행에는 영향이 없습니다.
#
# ■ 파일 전체 구조 (위에서 아래로 읽으면 됩니다)
#
#   0.  상수           창 크기, 색(BGR), 샤프닝 커널, 피부색 HSV 범위, 손 관절 연결 정보
#   1.  공통 유틸       글씨 그리기(put / outlined), 영상 크기·형식 통일(to_view / fit_frame),
#                      한글 글씨 렌더러(TextRenderer)
#   2.  카메라 입력      FrameSource: Jetson GStreamer(USB) / 영상 파일 / 이미지 / 가짜(demo) 영상
#   3.  효과 함수       p_* = 영상을 일부러 "망가뜨리는" 함수 (= 문제)
#                      s_* = 보기로 나오는 "해결 후보" 함수
#                      metric_* = 선명도 / 노이즈 / 대비를 숫자로 재는 함수
#   4.  build_questions()  기초 퀴즈 8문제의 데이터 (문제 문장, 보기 5개, 정답, 해설)
#   4b. 단계 미션         앞 단계의 정답 결과가 다음 단계의 입력이 되는 파이프라인 문제 5개
#                      (Kalman 필터 / MediaPipe 보조 함수 포함)
#   5.  Quiz 클래스       메뉴·문제·정답확인·결과 요약 화면의 상태 머신 + 그리기 + 마우스/키 처리
#   6.  HandTracker      MediaPipe 손 인식 래퍼 (모델 파일이 없으면 "사용 불가"로 처리)
#   7.  FreePlay         퀴즈 없이 키보드로 필터를 켜고 끄며 실험하는 자유 실습 모드
#   8.  App / main       창 만들기, 확대 배율(scale), 마우스 콜백, 메인 루프, 명령행 인자
#
# ■ 한 프레임이 화면에 나오기까지 (가장 중요한 흐름)
#
#   카메라 프레임(원본)
#     -> (문제가 있으면) p_* 로 망가뜨림           ... "문제 화면"
#     -> (보기를 골랐으면) 그 보기의 s_* 를 적용     ... "결과 화면"
#     -> to_view() 로 항상 640x480 BGR uint8 로 통일
#     -> 오른쪽 460px 패널(문제 / 보기 / 해설)을 옆에 붙여 1100x480 "논리 캔버스" 완성
#     -> App.present() 가 확대 배율(scale)을 곱해서 imshow
#
# ■ 용어
#   - BGR      : OpenCV 의 기본 채널 순서 (파랑, 초록, 빨강). matplotlib/skimage 는 RGB.
#   - 마스크    : 0 또는 255 로만 이루어진 1채널 흑백 영상 (255 = 선택된 픽셀).
#   - 논리 캔버스: 배율을 곱하기 전의 1100x480 좌표계. 버튼 위치/마우스 좌표는 전부 이 좌표계 기준.
#   - 미션 / 단계(stage): 한 문제 안에서 여러 단계를 차례로 푸는 문제 묶음.
#
# ■ 이 파일에 나오는 OpenCV / NumPy 함수 치트시트  (오른쪽 괄호 = 이 파일에서 쓰이는 곳)
#   cv2.flip(img, code)                  code 1 = 좌우, 0 = 상하, -1 = 둘 다                       (p_flip_ud, s_flip*)
#   cv2.cvtColor(img, 코드)               색공간 변환: BGR2RGB / BGR2GRAY / BGR2HSV / BGR2LAB      (거의 모든 곳)
#   cv2.blur(img, (k,k))                 box 블러(단순 평균)                                       (s_box*)
#   cv2.GaussianBlur(img, (k,k), 0)      가우시안 블러. k 는 홀수, sigma=0 이면 자동 계산           (s_gauss*, p_blur)
#   cv2.filter2D(img, -1, kernel)        임의 커널 합성곱(샤프닝 등). -1 = 입력과 같은 자료형       (s_sharp)
#   cv2.Canny(gray, lo, hi)              엣지 검출. uint8 입력, lo < hi (히스테리시스 임계값)        (s_canny_*)
#   cv2.Sobel + cv2.magnitude            x/y 방향 1차 미분과 그 크기                               (FreePlay 엣지)
#   cv2.Laplacian(gray, cv2.CV_32F)      2차 미분. 분산(var)으로 선명도 측정                       (metric_sharp)
#   cv2.inRange(hsv, lo, hi)             범위 안이면 255, 밖이면 0 인 마스크 생성                  (skin_mask)
#   cv2.bitwise_and(a, b, mask=m)        마스크가 255 인 곳만 남김.  bitwise_or 는 합집합           (apply_mask, with_specks)
#   cv2.getStructuringElement            모폴로지용 커널(타원/사각형)                              (_K5)
#   cv2.morphologyEx(m, OP, k)           OPEN(침식->팽창), CLOSE(팽창->침식) 등                     (s_open, s_close)
#   cv2.erode / cv2.dilate               침식(흰 영역 축소) / 팽창(흰 영역 확대)                    (s_erode4, s_dilate)
#   cv2.createCLAHE(...).apply(L)        국소 히스토그램 평활화. 색은 두고 밝기 채널(L)에만         (clahe_on_L)
#   cv2.equalizeHist(ch)                 전역 히스토그램 평활화 (1채널 uint8)                      (s_eq_bgr)
#   cv2.convertScaleAbs(f, alpha, beta)  |alpha*f + beta| 를 0~255 로 포화 (밝기/대비 조절)         (p_dark)
#   cv2.cornerHarris(g32, 2, 3, 0.04)    코너 응답 지도. 입력은 1채널                              (_harris)
#   cv2.findContours(mask, mode, method) 윤곽선 목록. 입력은 1채널 마스크                          (_contours)
#   cv2.contourArea / moments / boundingRect / minEnclosingCircle   윤곽의 면적 / 모멘트 / 사각형 / 최소 원
#   cv2.KalmanFilter(4, 2)               predict() 로 예측, correct(z) 로 측정 보정                 (미션4, FreePlay)
#   cv2.putText / line / circle / rectangle / polylines / arrowedLine / drawMarker / drawContours   그리기 함수
#   cv2.resize(img, (w,h), interpolation) 크기 변경. 축소는 INTER_AREA, 확대는 INTER_LINEAR 가 무난
#   cv2.addWeighted(a, 1-t, b, t, 0)     두 영상 섞기(페이드)                                      (_video)
#   np.clip(x, 0, 255).astype(np.uint8)  범위를 자르고 uint8 로 변환 (오버플로 방지)
######################################################################
import argparse  # 명령행 옵션(--demo, --scale ...) 해석
import glob  # 한글 폰트 파일을 패턴으로 찾을 때 사용 (find_korean_font)
import math  # sin/cos(가짜 영상의 움직임), sqrt/pi(노이즈 추정)
import os  # 파일 경로 / 환경변수 (폰트, 손 모델 파일 찾기)
import platform  # OS 구분 (Linux 에서만 DISPLAY 검사 / xrandr 로 화면 크기 읽기)
import random  # 노이즈 패턴 무작위 선택, 보기 순서 섞기
import re  # 예외 메시지에서 필요 없는 꼬리 떼어 내기 (short_err)
import sys  # 프로그램 종료 코드(sys.exit)
import time
from collections import deque  # deque(maxlen=N): 최근 N개만 남기는 큐 -> 궤적(trail) 저장용
from functools import lru_cache  # 같은 계산을 매 프레임 반복하지 않도록 결과를 기억해 두는 데코레이터 (fit_lines)

import cv2  # OpenCV (이 파일은 4.x / 5.x 모두에서 동작하도록 확인했음)
import numpy as np  # 영상은 곧 numpy 배열: (높이, 너비, 채널) 형태의 uint8

# Pillow(PIL)는 "한글 글씨"를 그리기 위해서만 쓴다.
# cv2.putText 는 ASCII(영문/숫자/기호)만 그릴 수 있어서 한글을 넣으면 '???' 로 깨진다.
# Pillow 가 없거나 한글 폰트를 못 찾으면 HAVE_PIL / TextRenderer.ko 가 False 가 되어
# 모든 안내문이 영어(en) 문구로 바뀐다 (프로그램은 그대로 동작).
try:
    from PIL import Image, ImageDraw, ImageFont
    HAVE_PIL = True
# ImportError 뿐 아니라 어떤 import 오류든 영어 UI 로 안전하게 후퇴한다
except Exception:  # Pillow 없으면 영어 UI로 동작
    HAVE_PIL = False

# ======================================================================
# 0. 상수
# ======================================================================
# [화면 레이아웃]  논리 캔버스 = 영상(왼쪽 640x480) + 패널(오른쪽 460x480) = 1100x480
#
#     0                640        1100
#     +-----------------+-----------+
#     |                 |           |   <- 높이 480
#     |   카메라 영상    |  문제/보기  |
#     |   (VID_W x VID_H)|  (PANEL_W)|
#     +-----------------+-----------+
#
# 카메라가 어떤 해상도로 들어오든 fit_frame() 이 640x480 으로 맞춰 주므로,
# 이 파일의 모든 좌표/크기 계산은 640x480 을 기준으로 해도 안전하다.
VID_W, VID_H = 640, 480            # 영상 영역 (카메라 입력은 항상 이 크기로 맞춤)
PANEL_W = 460                      # 오른쪽 패널 폭
WIN_W, WIN_H = VID_W + PANEL_W, VID_H
# 창 이름. 같은 이름으로 imshow / setMouseCallback / getWindowProperty 를 호출해야 같은 창이 된다
WIN_NAME = "AirBall Vision Quiz Lab (06_03 Jetson)"
FADE_SEC = 0.5                     # 정답일 때 망가진 화면 -> 복원 화면 페이드 시간

# MediaPipe HandLandmarker 모델 파일의 레포 기준 상대 경로.
# HandTracker._find_model() 이 [--hand-model / 현재 폴더 기준 / 이 파일(01_project) 기준 / 한 단계 위(= 레포 루트)] 순서로 찾는다.
# 그래서 레포 루트에서 python 01_project/06_03_jetson.py 로 실행해도, 01_project 안에서 python 06_03_jetson.py 로 실행해도 된다.
# (모델은 pip 패키지에 들어 있지 않고 별도의 .task 파일이라서 따로 받아야 한다.)
HAND_MODEL = "src/models/MediaPipe/hand_landmarker.task"

# 기존 파이프라인은 한 글자도 바꾸지 않았다 (Jetson USB 카메라용, 검증된 설정).
# 05.py / 06.py 와 같은 문자열이다.
# 읽는 순서:  v4l2src(USB 카메라) -> YUY2 640x480 30fps 로 받음 -> videoconvert 로 BGR 변환
#             -> queue(가득 차면 오래된 프레임 버림) -> appsink(OpenCV 가 가져가는 출구)
# 핵심 옵션: 'drop=true max-buffers=1' = 항상 "가장 최신 프레임 1장"만 유지해서
#            처리가 느려도 영상이 점점 밀리는(지연 누적) 현상을 막는다.
GST_USB = (
    'v4l2src device=/dev/video0 ! '
    'video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! '
    'videoconvert ! video/x-raw,format=BGR ! '
    'queue leaky=downstream max-size-buffers=1 '
    'max-size-bytes=0 max-size-time=0 ! '
    'appsink drop=true max-buffers=1 sync=false'
)
# 노트북 J단원의 CSI 카메라 파이프라인
# Jetson CSI(리본 케이블) 카메라용 파이프라인. 노트북 J단원과 같다.
# nvarguscamerasrc 는 NVIDIA 전용 요소이므로 Jetson 에서만 동작한다.
# (memory:NVMM = GPU 메모리 버퍼; nvvidconv 가 일반 메모리 BGRx 로 옮겨 준다)
GST_CSI = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

FONT = cv2.FONT_HERSHEY_SIMPLEX  # cv2.putText 에 쓰는 기본 폰트 (ASCII 전용, 한글 불가)
# 색 (BGR)
# 주의: OpenCV 색은 (B, G, R) 순서이다. (255, 0, 0) 은 빨강이 아니라 파랑!
C_PANEL = (30, 28, 26)  # 패널 배경색 (어두운 회색)
C_HEADER = (58, 50, 44)  # 패널 맨 위 제목 줄 배경
C_BTN = (56, 52, 48)  # 보기 버튼 기본색
C_BTN_HOVER = (92, 82, 70)  # 마우스를 올렸을 때의 버튼색
C_BTN_WRONG = (45, 45, 150)  # 오답으로 눌린 버튼 (붉은 계열)
C_BTN_RIGHT = (60, 140, 60)  # 정답 버튼 (초록 계열)
C_TEXT = (238, 238, 238)  # 기본 글자색 (거의 흰색)
C_DIM = (160, 160, 160)  # 보조 글자색 (흐린 회색)
C_GOOD = (90, 210, 90)  # '정답 / 좋음' 표시색
C_BAD = (80, 80, 245)  # '오답 / 나쁨' 표시색
C_WARN = (0, 200, 255)  # 경고 / 강조색 (노랑-주황)
C_ACCENT = (255, 190, 60)  # 제목 / 포인트색 (하늘색 계열)

# 샤프닝(선명화) 커널 3종.  filter2D 는 커널을 영상 위에 올려 "곱하고 더하는" 합성곱(상관) 연산을 한다.
#   SHARP_K  : 노트북에 나온 기본 커널. 중심 5, 상하좌우 -1.   (합 = 5 - 4 = 1)
#   SHARP_K8 : 이 퀴즈의 정답 커널. 중심 9, 이웃 8개 모두 -1.  (합 = 9 - 8 = 1)  <- 더 강하게 선명해짐
#   SHARP_KW : 약한 버전. 중심 3, 상하좌우 -0.5.               (합 = 3 - 2 = 1)
# 커널 원소의 합이 1 이면 "평평한 영역의 밝기"는 변하지 않고, 이웃과 차이가 나는 곳(엣지)만 강조된다.
# (합이 1보다 크면 전체가 밝아지고, 1보다 작으면 어두워진다.)
SHARP_K = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
SHARP_K8 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], np.float32)
SHARP_KW = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]], np.float32)

# 피부색 HSV 범위 [하한 ~ 상한].  OpenCV 8비트 HSV 의 범위는 H: 0~179, S: 0~255, V: 0~255.
#   H(색상) 0~20  : 빨강~주황~살구색 영역 (원래 0~360도를 반으로 줄여서 저장하므로 179가 최대)
#   S(채도) >= 40 : 너무 흐린(회색/흰색에 가까운) 픽셀 제외
#   V(밝기) >= 70 : 너무 어두운 픽셀 제외
# 주의: 이 범위는 BGR 영상이 아니라 "HSV 로 바꾼 영상"에만 써야 한다. (s_skin_bgr 가 일부러 보여 주는 함정)
# 조명/피부톤에 따라 달라지므로 완벽한 값이 아니라 '출발점'이다.
SKIN_LO = np.array([0, 40, 70], np.uint8)
SKIN_HI = np.array([20, 255, 255], np.uint8)

# MediaPipe 손 랜드마크 21개의 번호와 뼈대 연결 (손 스켈레톤을 선으로 그릴 때 사용).
# 번호 의미: 0 = 손목 / 1~4 = 엄지 / 5~8 = 검지 / 9~12 = 중지 / 13~16 = 약지 / 17~20 = 새끼
# (각 손가락의 4번째 번호가 손가락 끝(tip): 4, 8, 12, 16, 20)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9),
    (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
# 각 손가락의 "굽힘 정도"를 재기 위한 관절 3점 (A, B, C). B 를 꼭짓점으로 하는 각도 ABC 를 계산한다.
#   엄지 (1,2,3) / 검지 (5,6,7) / 중지 (9,10,11) / 약지 (13,14,15) / 새끼 (17,18,19)
# 손가락을 쭉 펴면 세 점이 거의 일직선 -> 각도 ~180도, 접으면 각도가 작아진다.
FINGER_ANGLE_POINTS = ((1, 2, 3), (5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19))
ANGLE_THRESHOLD = 160  # 노트북 M단원: 각도 >= 160 이면 손가락이 펴진 것

# CLAHE 객체는 한 번만 만들어 두고 재사용한다 (프레임마다 새로 만들면 낭비).
#   clipLimit=3.0     : 히스토그램 막대가 너무 높아지지 않게 자르는 한도 -> 노이즈가 과하게 증폭되는 것을 막음
#   tileGridSize=8x8  : 영상을 8x8 칸으로 나눠 칸마다 따로 평활화(= '국소' 대비 향상)
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


# ======================================================================
# 1. 공통 유틸
# ======================================================================
# cv2.putText 를 짧게 쓰기 위한 도우미.
#   org   : 글자 "왼쪽 아래" 기준점 (x, y)  <- 왼쪽 위가 아니라는 점에 주의
#   scale : 글자 크기 배율 (Hershey 폰트는 1.0 일 때 글자 높이가 대략 20px 안팎)
#   LINE_AA : 안티앨리어싱(가장자리를 부드럽게). 속도는 조금 느려지지만 글씨가 훨씬 깔끔하다.
# ASCII 문자열만 가능. 한글은 TextRenderer.draw() 를 써야 한다.
def put(img, text, org, scale=0.5, color=C_TEXT, thick=1):
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


# 영상 위에 올리는 글씨는 배경이 밝거나 어두운 곳을 오가므로 가독성을 위해 "검은 테두리"를 두른다.
# 방법: 같은 글씨를 검정으로 8방향(상하좌우 + 대각선 4개)으로 r 픽셀씩 밀어서 먼저 그린 뒤,
#       그 위에 원래 색 글씨를 그린다.
def outlined(img, text, org, scale=0.6, color=C_TEXT, thick=2):
    """검은 테두리 글씨. 테두리와 본문을 같은 굵기로 그려야 OpenCV 버전과 상관없이 글자가 어긋나지 않는다
    (굵기를 다르게 그리면 일부 버전에서 글자 간격이 달라져 겹쳐 보인다)."""
    x, y = org
    # 얇은 글씨(thick=1)는 1px, 굵은 글씨는 2px 만큼 테두리를 둔다
    r = 1 if thick < 2 else 2
    # 8방향: 좌, 우, 위, 아래, 좌상, 우상, 좌하, 우하
    for dx, dy in ((-r, 0), (r, 0), (0, -r), (0, r), (-r, -r), (r, -r), (-r, r), (r, r)):
        cv2.putText(img, text, (x + dx, y + dy), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


# ASCII 글씨 s 를 scale 배율로 그렸을 때의 가로 폭(px). put() 과 같은 폰트/굵기(1)로 잰다.
def text_w(s, scale):
    return cv2.getTextSize(s, FONT, scale, 1)[0][0]


# 코드 문자열 한 줄을 "max_w 픽셀 폭" 안에 1~2줄로 맞춘다.   반환 = (줄 튜플, 글자 배율)
#   textwrap.wrap 처럼 '글자 수'로 자르면 글자마다 폭이 달라 들쭉날쭉하고, 줄 끝에 `255))` 같은 토큰 찌꺼기가 남거나
#   [:2] 로 뒷부분이 잘릴 수 있다. 그래서 실제 픽셀 폭(text_w)을 재서 맞춘다.
#   1) scales 를 큰 것부터 보면서 한 줄에 들어가는 첫 배율을 쓴다.
#   2) 한 줄로는 안 되면 쉼표 뒤 / 공백 앞에서 두 줄로 나눈다.
#        - 괄호 깊이가 얕은 곳을 우선한다 (함수 인자 한가운데보다 바깥쪽에서 끊는 편이 읽기 좋다).
#        - 깊이가 같으면 두 줄 중 긴 쪽이 더 짧은 곳(= 균형 잡힌 곳)을 고른다.
#        - 깊이 1 이하로 끊을 수 있는 가장 큰 배율을 쓰고, 그런 배율이 없으면 가장 얕은 깊이로 끊는다.
#      둘째 줄 앞에는 indent 를 붙인다 (예: 코드는 공백 2칸, '#' 주석은 '#   ').
#   3) 그래도 안 들어가면 글자 수 기준으로 억지로 자른다 (아무 글자도 버리지 않는다. audit 이 이 경우를 잡아낸다).
# 같은 입력이 매 프레임 반복되므로 lru_cache 로 결과를 기억한다 (인자가 모두 해시 가능해야 해서 scales 는 튜플).
@lru_cache(maxsize=1024)
def fit_lines(text, max_w, scales, indent="", max_lines=2):
    # 1) 한 줄
    for sc in scales:
        if text_w(text, sc) <= max_w:
            return (text,), sc
    # 2) 두 줄: 끊을 수 있는 위치(head, tail, 깊이)를 모은다
    if max_lines >= 2:
        cands, depth = [], 0
        for i, c in enumerate(text):
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
            if c == "," and i < len(text) - 1:            # 쉼표는 쉼표 '뒤'에서 끊는다
                cands.append((text[:i + 1], text[i + 1:].lstrip(), depth))
            elif c == " " and i > 0 and text[i - 1] != ",":   # 공백은 그 '앞'에서 끊는다 (쉼표 뒤 공백은 위에서 처리됨)
                cands.append((text[:i].rstrip(), text[i + 1:].lstrip(), depth))
        fallback = None                                    # (깊이, 배율 순번, 배율, 줄들)
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
    # 3) 억지로 자르기
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


# 보기가 낸 예외를 화면에 한 줄로 보여 줄 짧은 문장으로 줄인다 (최대 limit 글자).
#   OpenCV 에러는 이런 모양이다:
#     "OpenCV(5.0.0) /io/opencv/modules/imgproc/src/canny.cpp:768: error: (-215:Assertion failed) _src.depth() == CV_8U in function 'Canny'"
#   앞의 버전/파일 경로는 학습에 도움이 안 되고, 그대로 자르면 정작 중요한 '이유'가 잘려 나간다. 그래서
#     1) 'error:' 앞(버전, 경로)을 버리고
#     2) findContours 처럼 진짜 이유가 '> ' 로 시작하는 줄에 따로 있으면 첫 번째 그 줄을 쓰고
#     3) '(-215:Assertion failed)' 꼬리표와 길기만 한 "in function '...'" 꼬리를 뗀다.   -> "_src.depth() == CV_8U"
#   OpenCV 가 아닌 에러(KeyError 등)는 그대로 한 줄로 합친다.
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
    # findContours 의 긴 안내문("Modes other than RETR_FLOODFILL and RETR_CCOMP support only ...")은 핵심만 남긴다
    s = re.sub(r"^Modes other than .*? support only", "findContours supports only", s)
    return ascii_text(s)[:limit]


# 글씨가 max_w 픽셀보다 길면 끝을 잘라 '...' 을 붙인다. 화면 폭을 넘어 글씨가 잘려 보이는 것을 막는다.
# (OpenCV 버전에 따라 같은 문장의 폭이 다르다. 예: 4.x 가 5.0 보다 넓게 나온다 -> 글자 수가 아니라 실제 폭을 재서 맞춘다)
def clip_text(s, max_w, scale):
    if text_w(s, scale) <= max_w:
        return s
    while len(s) > 1 and text_w(s + "...", scale) > max_w:
        s = s[:-1]
    return s + "..."


# cv2.putText 는 ASCII 만 그릴 수 있다. 예외 메시지에는 한글 경로 같은 비ASCII 글자가 섞일 수 있어서(한국어 Windows 등),
# 화면에 그리기 전에 비ASCII 글자를 '?' 로 바꾼다 (안 바꾸면 글자 한 개가 '???' 로 깨져 보인다).
def ascii_text(s):
    return str(s).encode("ascii", "replace").decode("ascii")


# 어떤 연산 결과가 나오든 "화면에 그릴 수 있는 표준 형태"로 바꾸는 안전장치.
# 표준 형태 = 640x480, BGR 3채널, uint8.
#   - None            -> 검은 화면
#   - float 등 uint8 아님 -> 0~255 로 자르고(clip) uint8 로 변환
#   - (H, W, 1) 형태     -> 마지막 축 제거해서 (H, W)
#   - (H, W) 흑백/마스크 -> GRAY2BGR 로 3채널화 (같은 값을 B, G, R 에 복사)
#   - 크기가 다르면      -> 640x480 으로 resize
# 오답 보기가 이상한 결과(흑백, 회전, 다른 크기 등)를 내놓아도 프로그램이 죽지 않고 "그 결과 그대로" 보이게 해 주는 핵심
#   함수.
def to_view(img):
    """어떤 결과든 640x480 BGR uint8 로 맞춘다 (회색 마스크, 다른 크기 포함)."""
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


# 마스크로 원본의 일부만 남기고 나머지는 검게 만든다. (inRange 결과를 눈으로 확인할 때 사용)
# cv2.bitwise_and(src, src, mask=m): 마스크가 255 인 픽셀만 원본 값을 유지하고 나머지는 0(검정).
def apply_mask(src_bgr, mask):
    """inRange 같은 마스크 결과를 원본에 적용해서 보여준다."""
    # 혹시 (H, W, 3) 마스크가 들어오면 첫 채널만 사용
    m = mask if mask.ndim == 2 else mask[:, :, 0]
    if m.dtype != np.uint8:
        # 0/1, bool, float 마스크도 0/255 uint8 로 통일 (bitwise_and 의 mask 인자는 uint8 이어야 함)
        m = (m > 0).astype(np.uint8) * 255
    if m.shape[:2] != src_bgr.shape[:2]:
        # 마스크는 최근접 보간으로 키워야 중간 회색값이 안 생긴다 (값은 0 또는 255 만 유지)
        m = cv2.resize(m, (src_bgr.shape[1], src_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.bitwise_and(src_bgr, src_bgr, mask=m)


# 카메라 프레임을 4:3 비율로 "중앙 크롭"한 뒤 640x480 으로 만든다.
# (그냥 resize 하면 16:9 영상이 옆으로 찌그러지므로, 비율이 다르면 먼저 잘라낸다)
#   w/h > 4/3 : 너무 가로로 김 -> 좌우를 잘라냄   (x0 = 남는 폭의 절반)
#   w/h < 4/3 : 너무 세로로 김 -> 위아래를 잘라냄 (y0 = 남는 높이의 절반)
# INTER_AREA : 축소할 때 가장 품질이 좋은 보간법 (모아레/계단 현상이 적음)
def fit_frame(f):
    """입력 프레임을 4:3 으로 중앙 크롭 후 640x480 으로 맞춘다."""
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


# 한글이 들어 있는 폰트 파일(.ttf/.otf/.ttc) 경로를 찾는다. 못 찾으면 None -> 영어 UI.
# 탐색 순서(앞쪽이 우선):
#   1) 환경변수 KO_FONT 로 직접 지정한 파일
#   2) 이 스크립트 옆의 ko_font.ttf / NanumGothic.ttf 등
#   3) Windows(맑은 고딕 등) / macOS 기본 폰트
#   4) Linux(Jetson 포함)의 /usr/share/fonts 아래 Nanum, Noto CJK 등  (sudo apt install fonts-nanum)
# 후보마다 ImageFont.truetype 으로 실제로 열어 보고, 열리는 첫 번째 파일을 돌려준다.
def find_korean_font():
    """한글이 나오는 폰트 파일을 찾는다. 없으면 None (영어 UI 로 동작)."""
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
        # '**' 패턴: 하위 폴더까지 재귀 검색. sorted 로 순서를 고정해 항상 같은 폰트가 선택되게 함
        cands += sorted(glob.glob(p, recursive=True))
    for c in cands:
        if c and os.path.isfile(c):
            try:
                ImageFont.truetype(c, 16)
                return c
            except Exception:
                continue
    return None


# 한글 / 영문 글씨를 "같은 방식"으로 그려 주는 클래스.
#
# ■ 왜 필요한가?  cv2.putText 는 한글을 못 그린다. 그래서 한글이 필요하면 Pillow 로 그린다.
# ■ 어떻게 합치나?  Pillow 로 '검은 바탕에 흰 글씨(L 모드, 0~255)' 이미지를 만들고, 이를 0~1 사이의
#    float32 "coverage(덮는 정도, 알파 마스크)"로 바꾼다. 그 다음 paste_cov() 가
#        결과 = 배경 * (1 - cov) + 글자색 * cov
#    로 BGR 영상에 섞는다. (글자 가장자리가 부드럽게 보이는 이유)
# ■ 성능: 문제 / 해설이 바뀔 때만 block() 을 호출하고 Quiz 쪽에서 결과(cov)를 캐시해서 쓴다.
#    그래서 매 프레임 Pillow 를 돌리지 않으며 FPS 에 영향이 거의 없다.
# ■ 한글 폰트가 없으면 self.ko = False -> 영문 모드: cv2.putText 로 같은 일을 흉내 낸다.
class TextRenderer:
    """한글(Pillow) / 영문(cv2) 텍스트를 같은 방식으로 그리는 도우미.
    문제가 바뀔 때만 호출해서 캐시하므로 FPS 에는 영향이 없다."""

    # lang='en' 이면 한글 폰트가 있어도 일부러 영어 UI 를 쓴다
    def __init__(self, lang="auto"):
        self.font_path = None
        if HAVE_PIL and lang != "en":
            self.font_path = find_korean_font()
        # True 이면 한글 UI 가능 (폰트 + Pillow 둘 다 있다는 뜻)
        self.ko = self.font_path is not None
        # 크기별 ImageFont 캐시 {글자크기: 폰트객체}. truetype() 로 파일을 여는 건 비싸서 한 번만 연다
        self._fonts = {}

    # size(px) 크기의 ImageFont 를 돌려준다. 처음 요청할 때만 파일을 열고 이후엔 캐시(self._fonts)에서 꺼낸다
    def font(self, size):
        if size not in self._fonts:
            self._fonts[size] = ImageFont.truetype(self.font_path, size)
        return self._fonts[size]

    # 문자열 s 를 size 크기로 그렸을 때의 가로 픽셀 폭 (줄바꿈 판단용)
    def _width(self, s, size):
        if self.ko:
            f = self.font(size)
            try:
                # Pillow 9.2+ : getlength 는 소수점까지 정확한 폭을 준다
                return f.getlength(s)
            except AttributeError:
                # 오래된 Pillow 에는 getlength 가 없어서(AttributeError) getsize 로 대체
                return f.getsize(s)[0]
        # 영문 모드: size(px) -> Hershey 배율로 환산(36px ≈ scale 1.0). getTextSize 의 [0][0] 이 폭
        return cv2.getTextSize(s, FONT, size / 36.0, 1)[0][0]

    # text 를 max_w 픽셀 폭 안에 들어가도록 줄바꿈해서 줄 목록(list[str])으로 돌려준다.
    # 규칙:
    #   1) '\n' 은 강제 줄바꿈 (문단 구분)
    #   2) 공백 기준으로 단어를 하나씩 붙여 보며 폭이 넘으면 줄을 바꾼다
    #   3) 단어 하나가 한 줄보다 길면 글자 단위로 자른다 (긴 코드 문자열 / 띄어쓰기 없는 한글 대비)
    def wrap(self, text, size, max_w):
        lines = []
        # para = 문단(강제 줄바꿈으로 나뉜 덩어리)
        for para in text.split("\n"):
            cur = ""
            for word in para.split(" "):
                # trial = 현재 줄에 이 단어를 붙여 본 후보 문자열
                trial = word if not cur else cur + " " + word
                if self._width(trial, size) <= max_w:
                    cur = trial
                    continue
                # 현재 줄에 단어를 더 못 붙이면, 지금까지 만든 줄(cur)을 확정하고 새 줄을 시작한다
                if cur:
                    lines.append(cur)
                    cur = ""
                if self._width(word, size) <= max_w:
                    cur = word
                else:  # 단어 하나가 너무 길면 글자 단위로 자른다
                    buf = ""
                    for ch in word:
                        if self._width(buf + ch, size) <= max_w:
                            buf += ch
                        else:
                            lines.append(buf)
                            buf = ch
                    cur = buf
            lines.append(cur)
        # 문단 끝에서도 마지막 줄을 추가했으므로(빈 문단이면 빈 줄 하나) '\n\n' 은 빈 줄로 표현된다
        return lines

    # 글 한 덩어리를 그려서 (coverage, 높이) 로 돌려준다.  coverage: float32 (h, w), 값 0~1.
    #   w         : 덩어리 폭(px)         size : 글자 크기(px)
    #   max_lines : 이 줄 수를 넘는 부분은 잘라 버린다 (패널 밖으로 넘치는 것 방지)
    # 줄 높이 lh = size * 1.4  (글자 높이의 1.4배 = 줄 간격 여유)
    def block(self, text, w, size, max_lines=None):
        """(coverage float32 HxW in 0..1, height) 반환. 흰 글씨 알파 마스크."""
        lines = self.wrap(text, size, w)
        if max_lines:
            lines = lines[:max_lines]
        # 줄 높이. 1.4 배는 한글이 위아래로 겹치지 않으면서 촘촘해 보이는 경험적 값
        lh = int(size * 1.4)
        h = max(lh * len(lines), 1) + 2
        if self.ko:
            # L = 8비트 흑백 모드. 검정(0) 바탕에 흰색(255)으로 글씨를 쓰면 그대로 알파 마스크가 된다
            img = Image.new("L", (w, h), 0)
            d = ImageDraw.Draw(img)
            f = self.font(size)
            for i, ln in enumerate(lines):
                d.text((0, i * lh), ln, font=f, fill=255)
            # PIL 이미지 -> numpy 배열(float32) 후 0~1 로 정규화
            cov = np.asarray(img, np.float32) / 255.0
        else:
            cov8 = np.zeros((h, w), np.uint8)
            for i, ln in enumerate(lines):
                # 영문 모드: putText 의 y 는 '글자 아래선' 이라서 줄 위치에 size 를 더해 준다
                cv2.putText(cov8, ln, (0, i * lh + size), FONT, size / 36.0, 255, 1, cv2.LINE_AA)
            cov = cov8.astype(np.float32) / 255.0
        return cov, h

    # 글 덩어리를 만들어 img 의 (x, y) 에 color 로 얹는다. 반환값 = 글이 끝난 다음 y (다음 요소를 이어 그릴 때 사용)
    def draw(self, img, text, x, y, w, size=16, color=C_TEXT, max_lines=None):
        cov, h = self.block(text, w, size, max_lines)
        paste_cov(img, cov, x, y, color)
        return y + h


# 알파 마스크 cov(0~1)를 img 의 (x, y) 위치에 color 색으로 섞어 넣는다. (img 를 직접 수정)
# 수식(픽셀마다):  새값 = 배경 * (1 - cov) + color * cov
#   - 화면 밖으로 삐져나가는 부분은 y2, x2 로 잘라낸다 (패널 경계 보호)
#   - float32 로 계산한 뒤 uint8 로 되돌린다
def paste_cov(img, cov, x, y, color):
    h, w = cov.shape[:2]
    y2, x2 = min(y + h, img.shape[0]), min(x + w, img.shape[1])
    if y2 <= y or x2 <= x:
        return
    # None 을 넣어 (h, w) -> (h, w, 1) 로 만들면 3채널(B, G, R)에 브로드캐스팅되어 같은 알파가 적용된다
    c = cov[:y2 - y, :x2 - x, None]
    sub = img[y:y2, x:x2].astype(np.float32)
    col = np.array(color, np.float32)
    img[y:y2, x:x2] = (sub * (1.0 - c) + col * c).astype(np.uint8)


# ======================================================================
# 2. 카메라 입력
# ======================================================================
# 카메라 없이도 모든 기능을 시험할 수 있게 만든 "가짜 영상" (--demo 옵션).
# t(초)에 따라 움직이는 물체들이 있다:
#   - 배경 그라디언트 (x 가 커질수록 파랑 증가, y 가 커질수록 초록 증가)
#   - 피부색 타원(얼굴 모양) : 사인 함수로 좌우/상하로 천천히 움직임 -> 피부색 마스크, Kalman 추적 실습용
#   - 초록 원, 파란 사각형, 'DEMO' 글자 : 엣지 / 코너 / 색 분할 실습용
def synthetic_frame(t):
    """카메라가 없을 때 쓰는 가짜 영상: 움직이는 피부색 타원 + 초록 원 + 배경."""
    f = np.zeros((VID_H, VID_W, 3), np.uint8)
    xs = np.linspace(0, 1, VID_W, dtype=np.float32)[None, :]
    ys = np.linspace(0, 1, VID_H, dtype=np.float32)[:, None]
    # B 채널: 왼쪽(60) -> 오른쪽(150) 으로 밝아지는 가로 그라디언트
    f[:, :, 0] = (60 + 90 * xs).astype(np.uint8)
    f[:, :, 1] = (70 + 80 * ys).astype(np.uint8)
    f[:, :, 2] = (60 + 40 * (1 - xs)).astype(np.uint8)
    cx = int(VID_W * (0.5 + 0.28 * math.sin(t * 1.3)))
    cy = int(VID_H * (0.5 + 0.22 * math.sin(t * 0.9 + 1.0)))
    # (140,170,230) = BGR. 즉 B=140, G=170, R=230 -> 살구색(피부색 HSV 범위에 들어감)
    cv2.ellipse(f, (cx, cy), (70, 90), 0, 0, 360, (140, 170, 230), -1, cv2.LINE_AA)   # 피부색
    cv2.circle(f, (cx - 25, cy - 20), 8, (40, 40, 40), -1, cv2.LINE_AA)
    cv2.circle(f, (cx + 25, cy - 20), 8, (40, 40, 40), -1, cv2.LINE_AA)
    gx = int(VID_W * (0.2 + 0.1 * math.cos(t * 0.7)))
    cv2.circle(f, (gx, 380), 40, (60, 200, 60), -1, cv2.LINE_AA)                      # 초록 공
    cv2.rectangle(f, (470, 330), (600, 440), (40, 90, 200), 3)
    cv2.putText(f, "DEMO", (20, 460), FONT, 1.0, (230, 230, 230), 2, cv2.LINE_AA)
    return f


# 카메라가 안 열릴 때 터미널에 보여 줄 안내문을 만든다. (확인할 수 있는 원인부터 차례로 적는다)
#   1) OpenCV 가 GStreamer 지원으로 빌드됐는가?  cv2.getBuildInformation() 안의 "GStreamer:  YES/NO" 줄을 읽는다.
#      pip 로 받은 opencv-python 은 보통 NO 이고, JetPack 에 들어 있는 OpenCV 는 YES 이다.
#   2) /dev/video0 장치 파일이 있는가?  없으면 USB 연결 문제이다.
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


# 영상 입력을 한 곳에 모은 클래스. 호출하는 쪽은 종류를 몰라도 read() 만 부르면 된다.
#   kind == 'camera' : Jetson USB 카메라 (GStreamer 파이프라인 GST_USB)
#   kind == 'video'  : 영상 파일 (끝나면 처음부터 반복 재생)
#   kind == 'image'  : 정지 이미지 한 장 (매 프레임 같은 그림)
#   kind == 'demo'   : synthetic_frame() 가짜 영상
# read() 는 항상 (성공여부, 640x480 BGR 프레임) 을 돌려준다.
class FrameSource:
    # 입력 종류 결정 우선순위: --demo > --image > --video > 카메라. 잘못된 경로면 바로 RuntimeError 로 이유를 알려 준다
    def __init__(self, args):
        self.kind = "camera"
        self.cap = None
        self.image = None
        # demo 영상의 시간 기준점. monotonic = 시스템 시계를 바꿔도 거꾸로 가지 않는 시계
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

    # 카메라를 여는 방법: 파일 위쪽의 GST_USB (또는 --csi 면 GST_CSI) 파이프라인을
    #   cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER) 로 연다. 05.py / 06.py 와 똑같은 방식이다.
    # 못 열면 camera_help() 가 만든 안내문(원인 후보)과 함께 RuntimeError 를 낸다.
    # 주의: isOpened() 가 True 여도 첫 프레임은 조금 늦게 올 수 있다 -> main 루프가 3초쯤 기다려 준다.
    @staticmethod
    def _open_camera(args):
        pipe = GST_CSI if args.csi else GST_USB
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(camera_help(args.csi))
        print("[camera] GStreamer pipeline opened (%s)" % ("CSI nvarguscamerasrc" if args.csi else "USB /dev/video0"))
        return cap

    # 프레임 한 장을 읽는다. 반환: (ok, frame). 실패하면 (False, None)
    def read(self):
        if self.kind == "demo":
            return True, synthetic_frame(time.monotonic() - self.t0)
        if self.kind == "image":
            # 원본 그림이 효과 함수에 의해 훼손되지 않도록 매번 복사본을 쓴다
            return True, fit_frame(self.image.copy())
        ok, f = self.cap.read()
        # 영상 파일 끝에 도달하면 0번 프레임으로 되감아 다시 읽는다 (무한 반복)
        if not ok and self.kind == "video":  # 영상 파일은 반복 재생
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = self.cap.read()
        if not ok:
            return False, None
        return True, fit_frame(f)

    # 카메라 / 영상 파일 자원을 반납한다 (main 의 finally 에서 호출)
    def release(self):
        if self.cap is not None:
            self.cap.release()


# ======================================================================
# 3. 효과 함수 모음 (문제로 망가뜨리는 함수 p_*, 보기로 쓰는 함수 s_*)
#    - 모두 BGR uint8 영상(또는 마스크)을 받아 결과를 돌려준다.
# ======================================================================
# 난수 생성기는 시드 7 로 고정 -> 실행할 때마다 같은 노이즈 모양이 나온다 (재현성).
# _NOISE / _SPECK 은 "미리 만들어 둔 패턴 풀(pool)". 처음 필요할 때 한 번만 만든다(지연 생성).
# 이유: 640x480x3 크기의 난수를 매 프레임 새로 만들면 느려서 FPS 가 떨어진다.
#       6개를 만들어 놓고 프레임마다 무작위로 하나 고르면 'TV 노이즈'처럼 자글거리면서도 빠르다.
_rng = np.random.default_rng(7)
_NOISE = []
_SPECK = []


# 가우시안 노이즈 패턴 6장. 평균 0, 표준편차 42 (값이 크므로 눈에 확 띔)
def _noise_pool():
    if not _NOISE:
        for _ in range(6):   # 프레임마다 다른 패턴을 골라 TV 노이즈처럼 자글거리게 한다
            # int16(부호 있는 정수)으로 저장하는 이유: 노이즈는 음수가 될 수 있다.
            # uint8 끼리 더하면 255 를 넘을 때 0 으로 되감기는(wrap-around) 문제가 있으므로,
            # int16 으로 더한 다음 clip(0~255) 하고 uint8 로 되돌린다 (p_noise 참고).
            _NOISE.append(_rng.normal(0, 42, (VID_H, VID_W, 3)).astype(np.int16))
    return _NOISE


# 마스크에 섞을 '작은 흰 점' 패턴 6장 (모폴로지 OPEN 문제용)
def _speck_pool():
    if not _SPECK:
        k = np.ones((2, 2), np.uint8)
        for _ in range(6):
            # 픽셀의 0.6% 를 무작위로 255(흰색)로 찍는다. (random < 0.006 이 True 인 곳)
            seeds = (_rng.random((VID_H, VID_W)) < 0.006).astype(np.uint8) * 255
            # 2x2 로 팽창시켜 점을 1px 이 아니라 2x2 크기로 키운다 (눈에 보이게). 이 점들은 5x5 OPEN 에서 사라진다
            _SPECK.append(cv2.dilate(seeds, k))
    return _SPECK


# 피부색 마스크: BGR -> HSV 로 바꾼 뒤 inRange 로 H 0~20, S>=40, V>=70 범위만 255 로 만든다.
# (색 판단은 반드시 HSV 에서: 조명이 달라져도 H(색상)는 비교적 안정적이기 때문)
def skin_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, SKIN_LO, SKIN_HI)


# 효과 차이는 눈으로만 판단하면 애매할 수 있어서, 화면 왼쪽 위에 "숫자"로도 보여 준다.
# 세 가지 지표: 선명도(Laplacian 분산) / 노이즈 크기(Immerkaer 추정) / 대비(밝기 표준편차)
# ---- 화면에 띄우는 측정값 (효과 차이를 숫자로 보여 준다) ----
# Immerkaer 노이즈 추정용 커널 N = [[1,-2,1],[-2,4,-2],[1,-2,1]].
# 이 커널은 "완만한 밝기 변화(평면, 경사)"에 대해서는 결과가 0 이 되도록 설계되어 있어서,
# 이 커널을 통과한 값에 남는 것은 대부분 '노이즈 성분'이다. -> 절댓값 평균으로 노이즈 크기를 추정.
_NOISE_K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


# BGR(3채널)이면 흑백으로 바꾸고, 이미 1채널이면 그대로 돌려준다 (지표 함수들이 둘 다 받을 수 있게)
def _gray_of(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# 선명도 = 라플라시안(2차 미분) 결과의 "분산".
# 라플라시안은 밝기가 급변하는 곳(엣지)에서 큰 값을 낸다. 선명한 영상은 엣지가 강해 분산이 크고,
# 흐린 영상은 변화가 완만해서 분산이 작다. (포커스 측정에 흔히 쓰는 방법)
# CV_32F: 음수 값을 잃지 않게 float32 로 계산 (uint8 로 하면 음수가 0 으로 잘림).
# ※ 노이즈도 엣지처럼 보이므로, 노이즈가 많으면 선명도가 '가짜로' 높게 나올 수 있다.
def metric_sharp(img):
    """선명도: 라플라시안 분산. 클수록 선명."""
    return float(cv2.Laplacian(_gray_of(img), cv2.CV_32F).var())


# 노이즈 크기 추정 (Immerkaer 방법).  sigma ≈ sqrt(pi/2) / 6 * mean(|I * N|)
#   - [1:-1, 1:-1] : 커널이 걸치지 못하는 바깥 1픽셀 테두리를 버린다
#   - BORDER_REFLECT : 가장자리는 반사해서 채움
# 값이 작을수록 깨끗한 영상.
def metric_noise(img):
    """노이즈 크기 추정(Immerkaer 방법). 작을수록 깨끗."""
    g = _gray_of(img).astype(np.float32)
    c = cv2.filter2D(g, -1, _NOISE_K, borderType=cv2.BORDER_REFLECT)[1:-1, 1:-1]
    return float(math.sqrt(math.pi / 2.0) / 6.0 * np.abs(c).mean())


# 대비 = 밝기(흑백 값)의 표준편차. 값이 넓게 퍼질수록(어둡고 밝은 곳이 섞일수록) 커진다
def metric_contrast(img):
    """대비: 밝기 표준편차. 클수록 대비가 큼."""
    return float(_gray_of(img).std())


# 이름 -> (표시 이름, 함수, 높을수록 좋은가)
# 지표 이름 -> (화면에 쓸 이름, 계산 함수, 높을수록 좋은가)
# 세 번째 값(higher) = '값이 클수록 좋은가'. 선명도·대비는 True, 노이즈는 False.
# Quiz._metric 이 '적용 전 -> 적용 후' 숫자를 비교해 좋아졌는지 판단할 때 쓴다
# (단, 초록색 표시는 숫자가 좋아졌더라도 '정답을 골랐을 때만' 쓴다: Canny 처럼 숫자만 높고 오답인 경우가 있어서)
METRICS = {
    "sharp": ("Sharpness", metric_sharp, True),
    "noise": ("Noise level", metric_noise, False),
    "contrast": ("Contrast", metric_contrast, True),
}


# p_* 함수 = "문제를 만드는" 함수. 깨끗한 프레임을 받아 일부러 망가뜨린다.
# 문제 화면은 이 함수의 출력이고, 정답 보기는 이 망가짐을 되돌리거나 해결해야 한다.
# ---- 문제(망가뜨리기) ----
# 상하 반전 (flipCode 0 = x축 기준 뒤집기). 같은 연산을 한 번 더 하면 원래대로
def p_flip_ud(f):
    return cv2.flip(f, 0)


# BGR -> RGB 로 채널 순서를 바꿔 버린다. 화면에서는 R/B 가 바뀌어 피부가 파랗게 보인다. 한 번 더 바꾸면 복원
def p_swap_rb(f):
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)


# 9x9 가우시안 블러로 흐리게 만든다. (sigma=0 이면 OpenCV 가 커널 크기에서 자동으로 계산)
def p_blur(f):
    return cv2.GaussianBlur(f, (9, 9), 0)      # 샤프닝 효과가 분명히 보이도록 꽤 흐리게


# 프레임마다 노이즈 풀에서 패턴을 하나 골라 더한다.
#   f(uint8) -> int16 으로 바꿔서 더함 -> 0~255 로 clip -> uint8 로 복귀
# (uint8 + int16 을 바로 더하면 오버플로/되감기가 생기기 때문에 이 순서를 지킨다)
def p_noise(f):
    pool = _noise_pool()
    n = pool[random.randrange(len(pool))]
    return np.clip(f.astype(np.int16) + n, 0, 255).astype(np.uint8)


# convertScaleAbs: 결과 = |alpha * 픽셀 + beta| 를 0~255 로 포화.
#   alpha 0.35 로 어둡게(곱하기), beta 10 으로 약간 들뜬 느낌(더하기)
def p_dark(f):
    return cv2.convertScaleAbs(f, alpha=0.35, beta=10)


# 피부색 마스크에 작은 흰 점 노이즈를 OR 로 합친다 (bitwise_or: 둘 중 하나라도 255 면 255)
def p_noisy_mask(f):
    pool = _speck_pool()
    return cv2.bitwise_or(skin_mask(f), pool[random.randrange(len(pool))])


# s_* 함수 = "보기(해결 후보)". 이름 규칙: s_ + 하는 일.  모두 입력 하나(f 또는 m)를 받아 결과를 돌려준다.
#   f = BGR 영상,  m = 마스크(1채널).
# 한 줄짜리 def 가 많은 이유: 보기 코드(문자열)와 실제로 실행되는 함수를 나란히 두고 비교하기 쉽게 하려는 것.
# ---- 보기(해결 후보) ----
# flipCode: 1 = 좌우 반전, 0 = 상하 반전, -1 = 상하+좌우 (= 180도 회전과 같음)
def s_flip0(f): return cv2.flip(f, 0)
def s_flip1(f): return cv2.flip(f, 1)  # 좌우 반전 (flipCode 1)
def s_flipm1(f): return cv2.flip(f, -1)  # 상하 + 좌우 동시 반전 (flipCode -1) = 180도 회전
# NumPy 슬라이싱으로 열(가로) 순서를 뒤집기 = 좌우 반전. 슬라이스는
#   view 라서 .copy() 로 연속 메모리 복사본을 만든다(OpenCV 함수에 안전)
def s_slice_lr(f): return f[:, ::-1].copy()
# 90도 회전: 가로/세로가 바뀐다 (480x640). to_view 가 640x480 으로 억지로 늘려 맞추므로 화면이 찌그러져 보인다
def s_rot90(f): return cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE)
# 채널 순서 교환. 결과는 여전히 uint8 3채널 배열이라 그냥 BGR 로 그려지고, 그래서 색이 바뀌어 보이거나(오답 상황) 원래대로
#   돌아온다(정답 상황)
def s_bgr2rgb(f): return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
def s_bgr2gray(f): return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)  # BGR -> 흑백(1채널). 색 정보가 사라진다
def s_bgr2hsv(f): return cv2.cvtColor(f, cv2.COLOR_BGR2HSV)  # BGR -> HSV (H 0~179). 결과를 BGR 로 그리면 색이 엉뚱하게 보인다
def s_bgr2lab(f): return cv2.cvtColor(f, cv2.COLOR_BGR2LAB)  # BGR -> LAB (L 밝기, a/b 색). 이것도 BGR 로 그리면 색이 이상하다
# GaussianBlur 의 커널 크기는 반드시 '홀수'. 클수록 더 많이 평균을 내므로 더 흐려진다
def s_gauss11(f): return cv2.GaussianBlur(f, (11, 11), 0)
def s_gauss9(f): return cv2.GaussianBlur(f, (9, 9), 0)  # 9x9: Q4 / 미션1 의 정답. 노이즈가 눈에 띄게 줄면서 형태는 남는 크기
def s_gauss21(f): return cv2.GaussianBlur(f, (21, 21), 0)  # 21x21: 아주 많이 흐려진다 (선명화 문제의 오답)
# cv2.blur = box filter(단순 평균). 가우시안과 달리 가까운 픽셀과 먼 픽셀에 같은 가중치를 준다
def s_box21(f): return cv2.blur(f, (21, 21))  # 21x21 box 블러: 가우시안보다 네모난 번짐이 남는다
# 1x1 커널 = 자기 자신만 곱함 = 아무 변화 없음 ('효과 없는 코드' 오답 보기용)
def s_gauss1(f): return cv2.GaussianBlur(f, (1, 1), 0)      # 1x1 커널: 아무것도 섞지 않음
# 정답 샤프닝. ddepth=-1 은 '입력과 같은 자료형(uint8)'으로 출력하라는 뜻. 결과는 자동으로 0~255 포화(saturate)
def s_sharp(f): return cv2.filter2D(f, -1, SHARP_K8)
# 3채널 BGR 에 바로 Canny 를 쓴 코드. 최신 OpenCV 는 3채널도 처리하지만
#   결과가 '흑백 엣지 영상'이라 선명화/노이즈제거의 답이 아니다
def s_canny_bgr(f): return cv2.Canny(f, 100, 200)


# 정답 Canny: 흑백(uint8 단일 채널)으로 바꾼 뒤 낮은 임계값 100, 높은 임계값
#   200. 200 이상=확실한 엣지, 100~200=이웃이 확실한 엣지와 이어져 있을 때만 엣지
def s_canny_gray(f):
    return cv2.Canny(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), 100, 200)


# Canny 에 float32 를 넣으면 에러. Canny 는 8비트(uint8) 입력만
#   지원한다 (노트북의 함정). 에러 메시지는 화면에 그대로 표시된다
def s_canny_f32(f):
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray.astype(np.float32) / 255., 100, 200)


# 흑백으로 바꾼 후 블러. 블러는 엣지를 뭉개는 연산이라 '엣지 추출'의 반대 방향
def s_blur_gray(f):
    return cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (11, 11), 0)


# 흑백 영상에 샤프닝. 엣지가 강조된 '흑백 영상 전체'가 나오지, 엣지만 따로 뽑히지는 않는다
def s_sharp_gray(f):
    return cv2.filter2D(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), -1, SHARP_K8)


# 정답 피부색 마스크: BGR -> HSV 변환 후 inRange. 결과는 1채널
#   마스크(0/255)이고, 표시는 '마스크 적용' 방식(apply_mask)으로 한다
def s_skin_hsv(f):
    return cv2.inRange(cv2.cvtColor(f, cv2.COLOR_BGR2HSV), SKIN_LO, SKIN_HI)


def s_skin_bgr(f):
    # 함정: HSV 용 범위(H 0~20, S 40~, V 70~)를 BGR 값에 그대로 적용하면
    #   파랑(B) 0~20, 초록(G) 40~255, 빨강(R) 70~255 인 픽셀이 선택된다.
    # 피부는 파랑 성분이 20보다 큰 경우가 많아서 오히려 빠지고, 엉뚱한 영역(순수한 노랑/빨강 계열)이 나온다
    return cv2.inRange(f, SKIN_LO, SKIN_HI)  # 일부러 틀린 코드: BGR 값에 HSV 범위를 사용


# H 범위 100~130 = 파랑 계열. HSV 에서 색상(H)을 고르는 방법은 맞지만 '피부색' 범위가 아니다
def s_blue_hsv(f):
    return cv2.inRange(cv2.cvtColor(f, cv2.COLOR_BGR2HSV),
                       np.array([100, 40, 70], np.uint8), np.array([130, 255, 255], np.uint8))


# threshold(…)[1]: threshold 는 (사용한임계값, 결과영상) 튜플을 돌려주므로 [1]
#   이 결과 영상. 밝기만 보고 127 기준으로 흑/백을 나눔 (색 정보 사용 안 함)
def s_thresh_gray(f):
    return cv2.threshold(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1]


# 모폴로지 연산용 5x5 타원형 구조 요소(커널).
# MORPH_ELLIPSE: 둥근 모양이라 MORPH_RECT 보다 윤곽이 부드럽게 처리된다.
# 크기 5x5: 2x2 짜리 점 노이즈는 완전히 지워지고, 큰 덩어리(얼굴/손)는 거의 그대로 남는 크기.
_K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# OPEN = 침식(erode) 후 팽창(dilate). 구조 요소보다 작은 흰 점은 침식에서
#   사라져서 팽창해도 되살아나지 않는다. 큰 덩어리는 원래 크기로 복원
def s_open(m): return cv2.morphologyEx(m, cv2.MORPH_OPEN, _K5)
# CLOSE = 팽창 후 침식. 흰 덩어리 안의 작은 검은 구멍을 메운다. (흰 점 노이즈 제거에는 효과 없음)
def s_close(m): return cv2.morphologyEx(m, cv2.MORPH_CLOSE, _K5)
# 팽창만 하면 흰 영역이 커지므로 점 노이즈도 같이 커진다 (돌이킬 침식이 없음)
def s_dilate(m): return cv2.dilate(m, _K5)
# 마스크에 블러를 걸면 0/255 의 이진 마스크가 회색 번짐으로 바뀐다 (점이 사라지지 않고 희미해질 뿐)
def s_gauss_mask(m): return cv2.GaussianBlur(m, (5, 5), 0)
# 마스크의 경계선만 뽑는다. 점 노이즈도 각각 작은 테두리로 남는다
def s_canny_mask(m): return cv2.Canny(m, 100, 200)


# 정답: LAB 색공간의 밝기 채널(L)에만 CLAHE 를 적용한다.
#   LAB: L = 밝기(Lightness), a / b = 색 성분(초록-빨강, 파랑-노랑)
#   밝기(L)만 평활화하면 색(a, b)은 그대로 두고 대비만 살릴 수 있다.
# 순서: BGR -> LAB -> split(L, a, b) -> L 만 CLAHE -> merge -> BGR 로 복귀
def clahe_on_L(f):
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _CLAHE.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# 오답: B, G, R 세 채널을 각각 equalizeHist.
# 채널마다 히스토그램을 따로 펴면 채널 간 비율(= 색)이 바뀌어서 색이 틀어진다.
# (그리고 equalizeHist 는 CLAHE 와 달리 영상 전체를 한꺼번에 평활화하므로 국소 대비는 못 살린다)
def s_eq_bgr(f):
    b, g, r = cv2.split(f)
    return cv2.merge([cv2.equalizeHist(b), cv2.equalizeHist(g), cv2.equalizeHist(r)])


# ======================================================================
# 4. 문제 은행
#    choices: (코드 문자열, 함수, 정답여부, 해설(ko), 해설(en), 표시 방식)
#    표시 방식 "mask" 는 결과 마스크를 원본에 적용해서 보여준다.
# ======================================================================
# 기초 퀴즈 8문제의 데이터를 만들어 돌려준다 (함수로 감싼 이유: 모듈 로드 시점이 아니라 필요할 때 생성).
#
# 각 문제는 dict 이고 키는 다음과 같다:
#   unit, unit_name : 노트북 단원 표시 (A 영상 조정 / B 색공간 / C 필터링 / D 엣지 / E 색 분할 / F 모폴로지 / G 영상 개선)
#   ko, en          : 문제 문장 (한글 / 영어)
#   setup           : 화면에 보여 줄 "이미 실행된 코드" 줄들 (보기가 쓰는 변수 gray, hsv, kernel 등을 설명)
#   problem         : 영상을 망가뜨리는 p_* 함수 (None 이면 원본 그대로 시작)
#   metric          : (선택) 화면에 띄울 숫자 지표 이름 ('sharp' / 'noise' / 'contrast')
#   choices         : 보기 5개의 리스트. 각 보기는 6-튜플
#                     (코드 문자열, 적용 함수, 정답 여부, 해설 ko, 해설 en, 표시 방식)
#                     표시 방식  'image' = 결과를 영상으로 그대로 보여줌
#                                'mask'  = 결과 마스크를 원본에 씌워(apply_mask) 보여줌
# 보기 5개의 순서는 Quiz 가 문제마다 섞어서 보여 준다 (정답이 항상 1번이 되지 않도록).
# 오답 보기는 '실제로 학생이 자주 하는 실수'(함정)와 '효과는 있지만 목적과 다른 함수'로 구성했다.
def build_questions():
    # 여러 문제에서 보기 코드 `sharp_kernel` 이 무엇인지 알려 주기 위한 공통 설명 줄
    # 주의: filter2D 의 커널은 반드시 numpy 배열(float32)이어야 한다. 파이썬 리스트를 그대로 넣으면 OpenCV 가 에러를 낸다
    sharp_setup = "# sharp_kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]], np.float32)"
    # 문제 dict 들을 차례로 담는 리스트 (A -> G 순서)
    qs = []

    # [문제 1 / 단원 A: 영상 조정]  상하 반전을 되돌리기
    #  - 문제: p_flip_ud 가 화면을 위아래로 뒤집어 놓는다.
    #  - 정답: cv2.flip(frame, 0)  (같은 뒤집기를 한 번 더 하면 원래대로)
    #  - 오답 설계: flipCode 를 헷갈리게 하는 1(좌우) / -1(둘 다), NumPy 슬라이싱 좌우 반전, 90도 회전.
    qs.append(dict(
        unit="A", unit_name="Image adjustment",
        ko="화면이 위아래로 뒤집혀 보입니다. 원래대로 돌리는 코드는?",
        en="The view is upside down. Which line restores it?",
        setup=[], problem=p_flip_ud,
        choices=[
            # 정답: flipCode 0
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

    # [문제 2 / 단원 B: 색공간]  R 과 B 채널이 바뀐 화면 바로잡기
    #  - 문제: p_swap_rb 가 BGR -> RGB 로 바꿔 놓아서 피부가 파랗게 보인다.
    #  - 정답: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  (채널 순서를 한 번 더 맞바꿈)
    #  - 오답 설계: 다른 색공간(GRAY / HSV / LAB) 변환 - '색공간 변환 함수니까 맞겠지'라는 착각 유도.
    #    (HSV, LAB 값을 BGR 처럼 그리면 색이 엉뚱해지는 모습을 눈으로 확인할 수 있다)
    qs.append(dict(
        unit="B", unit_name="Color space",
        ko="피부가 파랗게 보입니다. 채널 순서(B, R)가 뒤바뀐 상태예요. 바로잡는 코드는?",
        en="Skin looks blue: the B and R channels are swapped. Which line fixes it?",
        setup=[], problem=p_swap_rb,
        choices=[
            # 정답
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

    # [문제 3 / 단원 C: 필터링 - 선명화]  흐릿한 영상을 선명하게
    #  - 문제: p_blur(9x9 가우시안 블러). metric='sharp' 라서 왼쪽 위에 '선명도' 숫자가 뜬다.
    #  - 정답: filter2D + 샤프닝 커널(SHARP_K8). 선명도 숫자가 크게 오른다 (실측: 약 13 -> 139).
    #  - 오답 설계: 블러류(21x21 가우시안 / 21x21 box)는 반대 방향이라 숫자가 더 떨어짐(약 2~3),
    #    Canny 는 윤곽선만 남기고, BGR2GRAY 는 선명도가 그대로.
    #  - 오답과 정답의 차이가 눈으로 분명하게 보이도록 오답 블러는 일부러 큰 커널(21)을 썼다.
    qs.append(dict(
        unit="C", unit_name="Filtering",
        ko="영상이 많이 흐릿합니다. 선명하게 만드는 코드는? (화면 왼쪽 위의 선명도 숫자도 같이 보세요)",
        en="The view is very blurry. Which line sharpens it? (watch the Sharpness number, top-left)",
        setup=[sharp_setup], problem=p_blur, metric="sharp",
        choices=[
            # 정답: 합이 1 인 샤프닝 커널 -> 평평한 곳 밝기 유지 + 엣지 강조
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

    # [문제 4 / 단원 C: 필터링 - 노이즈 제거]  자글거리는 노이즈 줄이기
    #  - 문제: p_noise (표준편차 42 의 강한 가우시안 노이즈). metric='noise' -> '노이즈 수치'가 뜬다.
    #  - 정답: GaussianBlur 9x9 (실측: 노이즈 수치 약 23.9 -> 0.3). 변화가 확실히 보이도록 커널을 크게 잡았다.
    #  - 오답 설계: 샤프닝(노이즈까지 증폭, 약 78) / Canny(점마다 엣지, 약 112) / 색공간 변환 / 좌우 반전(무관).
    qs.append(dict(
        unit="C", unit_name="Filtering",
        ko="화면이 자글자글한 노이즈로 가득합니다. 노이즈를 줄이는 코드는? (왼쪽 위 노이즈 수치도 같이 보세요)",
        en="The view is full of grainy noise. Which line reduces it? (watch the Noise number, top-left)",
        setup=[sharp_setup], problem=p_noise, metric="noise",
        choices=[
            # 정답: 가우시안 가중 평균 -> 무작위 노이즈가 평균되어 줄어듦 (대신 약간 흐려짐)
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

    # [문제 5 / 단원 D: 엣지]  경계선만 추출하기
    #  - 문제: 망가뜨리지 않은 원본(problem=None)에서 시작. setup 에 gray 변수가 이미 있다고 보여 준다.
    #  - 정답: cv2.Canny(gray, 100, 200)
    #  - 오답 설계: Canny 에 float32 를 넣는 함정(에러 발생), 블러(엣지를 지움), 색공간 변환, 샤프닝(전체가 나옴).
    qs.append(dict(
        unit="D", unit_name="Edges",
        ko="경계선(엣지)만 뽑아내고 싶습니다. 올바른 코드는?",
        en="Extract only the edges. Which code is right?",
        setup=["# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)", sharp_setup], problem=None,
        choices=[
            # 정답: uint8 단일 채널 + 두 임계값
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

    # [문제 6 / 단원 E: 색 분할]  피부색만 남기기
    #  - 문제: 원본에서 시작 (카메라에 얼굴/손을 보여 줘야 결과가 보인다).
    #  - 정답: cv2.inRange(hsv, (0, 40, 70), (20, 255, 255))  -> 표시 방식 'mask' (원본에 마스크 적용)
    #  - 오답 설계: BGR 에 HSV 범위를 쓰는 함정 / 파랑 범위 / 밝기 이진화 / Canny.
    #  - hsv 변수는 setup 줄에서 '이미 변환되어 있다'고 가정한다.
    qs.append(dict(
        unit="E", unit_name="Color segmentation",
        ko="화면에서 내 피부색(얼굴·손)만 남기고 싶습니다. 올바른 코드는? (카메라에 얼굴이나 손을 보여 주세요)",
        en="Keep only skin-colored pixels (face, hands). Which code is right? (show your face or hand)",
        setup=["# hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)",
               "# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)",
               "# mask is applied: cv2.bitwise_and(frame, frame, mask=mask)"], problem=None,
        choices=[
            # 정답: 색 판단은 HSV 에서
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

    # [문제 7 / 단원 F: 모폴로지]  마스크의 작은 흰 점 노이즈 제거
    #  - 문제: p_noisy_mask = 피부색 마스크 + 2x2 흰 점들.
    #  - 정답: MORPH_OPEN (침식 후 팽창)  -> 점은 사라지고 큰 덩어리는 유지.
    #  - 오답 설계: CLOSE(구멍 메우기 - 방향이 반대), dilate(점이 커짐), 블러(번짐), Canny(테두리).
    qs.append(dict(
        unit="F", unit_name="Morphology",
        ko="피부색 마스크에 작은 흰 점 노이즈가 가득합니다. 점은 지우고 큰 덩어리는 살리는 코드는?",
        en="The skin mask is full of tiny white specks. Which line removes them but keeps the big blob?",
        setup=["# mask = skin mask + white speck noise",
               "# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))"], problem=p_noisy_mask,
        choices=[
            # 정답
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

    # [문제 8 / 단원 G: 영상 개선]  어둡고 뿌연 화면의 대비 살리기
    #  - 문제: p_dark (밝기 35% + 약간의 오프셋). metric='contrast' 라서 '대비' 숫자가 뜬다 (실측 약 26 -> 48).
    #  - 정답: LAB 의 L 채널에만 CLAHE -> 색은 유지한 채 대비 향상.
    #  - 오답 설계: BGR 각 채널을 따로 평활화(색 틀어짐), 블러, 흑백 변환, HSV 변환.
    qs.append(dict(
        unit="G", unit_name="Enhancement",
        ko="화면이 어둡고 뿌옇습니다. 색은 유지하면서 대비를 살리는 방법은?",
        en="The view is dark and flat. Which approach restores contrast while keeping colors?",
        setup=["# clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))",
               "# L, a, b = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))"],
        problem=p_dark, metric="contrast",
        choices=[
            # 정답. setup 에서 LAB 로 나눈 L, a, b 중 L 에만 CLAHE 를 적용하고 다시 합쳐 BGR 로 되돌린다 (clahe_on_L 과 같은 동작)
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
    # 문제 8개가 담긴 리스트 반환. Quiz 가 이것과 build_missions() 의 결과를 합쳐서 사용한다
    return qs


# ======================================================================
# 4b. 단계 미션 (여러 단계를 이어서 푸는 파이프라인 문제)
#     한 단계의 정답 결과가 다음 단계의 입력이 된다.
#     단계 보기 함수: fn(st, ctx) -> st'
#       st  : dict(frame=BGR 작업 영상, mask=마스크|None, vis=화면에 보일 영상|None,
#                  meas=(x, y)|None, hand=손 랜드마크 목록|None, pts=픽셀 좌표 목록|None)
#       ctx : 보기마다 따로 유지되는 저장소 (Kalman처럼 상태가 필요한 단계용, ctx["now"] = 현재 시각)
# ======================================================================
# 상태 dict 를 "복사한 뒤 일부만 바꿔서" 새 dict 로 돌려준다 (함수형 스타일; 원본 st 는 건드리지 않음).
# 왜 복사하나?  단계 함수들은 프레임마다 처음부터 다시 계산되는데, 같은 st 를 여러 보기가 공유한다.
#               한 보기가 st 를 직접 고치면 다른 보기/다음 프레임의 입력이 오염된다.
# 사용 예:  S(st, vis=img)  ->  vis 만 img 로 바뀐 새 상태
def S(st, **kw):
    d = dict(st)
    d.update(kw)
    return d


# 미션 파이프라인의 "초기 상태"를 만든다.  각 필드의 의미:
#   raw   : 카메라 원본 프레임 (망가뜨리기 전)
#   frame : 지금 단계의 작업 영상 (BGR 3채널). 단계가 진행될수록 뒤집기 / 노이즈 제거 등이 누적된다
#   mask  : 마스크 단계에서 생기는 1채널 흑백 영상 (없으면 None)
#   vis   : 화면에 '보여 줄' 영상. None 이면 frame 을 그대로 보여준다 (예: 윤곽선 그림은 frame 은 두고 vis 만 바꿈)
#   meas  : 측정된 위치 (x, y) 또는 None  <- Kalman 단계의 입력
#   hand  : MediaPipe 가 찾은 손 랜드마크 목록 또는 None
#   pts   : 랜드마크를 픽셀 좌표로 바꾼 목록 또는 None
def st_new(raw, frame):
    return dict(raw=raw, frame=frame, mask=None, vis=None, meas=None, hand=None, pts=None)


# 화면에 그릴 영상 = vis 가 있으면 vis, 없으면 frame (둘 다 to_view 로 표준 형태 보장)
def vis_img(st):
    v = st.get("vis")
    return to_view(v if v is not None else st["frame"])


# 단계 함수 만들기 (1): "영상 -> 영상" 함수 fn 을 (st, ctx) -> st 단계 함수로 포장한다. (클로저)
# 결과가 새 작업 영상(frame)이 되어 다음 단계로 이어진다. vis=None 으로 초기화해서 화면에도 이 결과가 보인다.
# 예) f_frame(s_flip0) = '작업 영상을 상하 반전하는 단계'
def f_frame(fn):
    """frame -> frame 함수를 단계 함수로 감싼다 (결과가 새 작업 영상이 된다)."""
    # 실제로 단계에서 호출되는 함수. 바깥의 fn 을 기억하고 있다 (클로저). ctx 는 이 단계에서는 쓰지 않는다
    def run(st, ctx):
        return S(st, frame=to_view(fn(st["frame"])), vis=None)
    return run


# 단계 함수 만들기 (2): 결과를 화면에만 보여 주고 작업 영상(frame)은 건드리지 않는다 (미리보기/오답 확인용)
def f_view(fn):
    """frame -> 결과(화면에만 보이고 작업 영상은 그대로)."""
    # frame 은 그대로 두고 vis(화면 표시용)만 바꾼 새 상태를 돌려준다
    def run(st, ctx):
        return S(st, vis=to_view(fn(st["frame"])))
    return run


# 마스크에 작은 흰 점을 섞는다. 미션2에서 '카메라 노이즈 때문에 마스크에 점이 생겼다'는 상황을 만들 때 사용
def with_specks(m):
    pool = _speck_pool()
    return cv2.bitwise_or(m, pool[random.randrange(len(pool))])


# 단계 함수 만들기 (3): "영상 -> 마스크" 함수. 결과에 흰 점 노이즈를 섞어서 mask 로 저장한다.
# → 다음 단계(모폴로지)에서 점을 제거하는 문제가 자연스럽게 이어진다.
def f_mask(fn):
    """frame -> 마스크. 이 미션에서는 카메라 노이즈로 작은 흰 점이 섞였다고 가정한다."""
    # fn(frame) 결과 마스크에 흰 점을 섞어서 mask 와 vis 에 함께 저장한다
    def run(st, ctx):
        m = with_specks(fn(st["frame"]))
        return S(st, mask=m, vis=to_view(m))
    return run


# 단계 함수 만들기 (4): 이미 만들어진 mask 에 연산(OPEN/CLOSE 등)을 적용해 mask 를 갱신한다
def f_maskop(fn):
    # 이전 단계에서 만든 mask 에 fn 을 적용한 결과를 다시 mask 로 저장한다
    def run(st, ctx):
        m = fn(st["mask"])
        return S(st, mask=m, vis=to_view(m))
    return run


# 윤곽선(contour) 목록을 돌려준다.
#   RETR_EXTERNAL    : 가장 바깥 윤곽선만 (구멍 안쪽 윤곽은 무시)
#   CHAIN_APPROX_SIMPLE : 직선 구간은 양 끝점만 저장해서 메모리 절약
# 입력은 반드시 "1채널 8비트" (0이 아닌 값 = 물체). 3채널 BGR 을 넣으면 에러.
# 반환 형태가 OpenCV 버전마다 다르다: 3.x 는 (image, contours, hierarchy) 3개, 4.x/5.x 는 (contours, hierarchy) 2개.
# -> len(res) 로 구분해서 contours 부분만 꺼낸다.
def _contours(mask):
    res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(res[0] if len(res) == 2 else res[1])


# Kalman 미션용: 화면에서 '피부색 덩어리 하나'를 찾는다.
# 순서: 피부색 마스크 -> MORPH_OPEN(점 제거) -> 윤곽선 -> 면적이 가장 큰 것.
# 면적 400 px² 미만이면 노이즈로 보고 None (= 측정 실패).
def _skin_blob(frame):
    """피부색 마스크 -> OPEN -> 가장 큰 윤곽 (너무 작으면 None)."""
    m = cv2.morphologyEx(skin_mask(frame), cv2.MORPH_OPEN, _K5)
    cs = _contours(m)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    return c if cv2.contourArea(c) >= 400 else None


# 그리기 함수에 넘길 정수 변환. 값이 이상하게 큰 경우(Kalman 발산 등) cv2 가 OverflowError 를 내지 않도록 ±30000 으로 제한
def _ci(v):
    """그리기용 정수 (너무 큰 값은 잘라서 OpenCV 오버플로를 막는다)."""
    return int(max(-30000, min(30000, v)))


# 미션에서 추가로 쓰는 보기 함수들 (오답 포함)
# ---- 추가 보기 함수 ----
# 180도 회전 = 상하+좌우 반전. 상하 반전만 되돌려야 하는 문제에서는 오답(좌우가 뒤집힘)
def s_rot180(f): return cv2.rotate(f, cv2.ROTATE_180)
# B 채널 하나만 꺼냄 -> (H, W) 흑백. '채널 분리'를 노이즈 제거로 착각하는 오답
def s_chan0(f): return f[:, :, 0].copy()
# 밝기를 절반으로 -> 더 어두워짐 (오답)
def s_dim(f): return cv2.convertScaleAbs(f, alpha=0.5)
# 임계값이 너무 낮음 -> 약한 변화(노이즈)까지 엣지로 잡혀 화면이 지저분해짐
def s_canny_lo(f): return cv2.Canny(_gray_of(f), 10, 30)
# 임계값이 너무 높음 -> 거의 모든 엣지가 사라져 화면이 비어 보임
def s_canny_hi(f): return cv2.Canny(_gray_of(f), 240, 255)
# 침식을 4번 반복 -> 점은 사라지지만 큰 덩어리도 많이 깎여 작아짐 (팽창으로 복원하는 OPEN 과의 차이)
def s_erode4(m): return cv2.erode(m, _K5, iterations=4)


# H 상한을 180 으로 너무 넓힌 피부색 범위 -> 거의 모든 색이 포함되어 마스크가 화면 대부분을 덮음
def s_skin_wide(f):
    return cv2.inRange(cv2.cvtColor(f, cv2.COLOR_BGR2HSV),
                       np.array([0, 40, 70], np.uint8), np.array([180, 255, 255], np.uint8))


# Harris 코너 검출 후 코너를 빨간색으로 표시한 영상을 돌려준다.
#   1) 흑백 -> float32 (cornerHarris 는 1채널 8비트 또는 float32 를 받는다)
#   2) cv2.cornerHarris(img, blockSize=2, ksize=3, k=0.04): 픽셀마다 '코너다운 정도' 응답값을 계산
#         blockSize = 이웃 창 크기 / ksize = 내부 Sobel 크기 / k = 경험 상수(0.04~0.06)
#   3) dilate: 응답 점을 조금 키워서 눈에 잘 보이게 함
#   4) 응답 > ratio * 최댓값 인 곳을 빨강으로 칠함.  ratio 가 임계값 역할이다:
#         0.01   (보통 쓰는 값)  : 코너가 적당히 표시됨   <- 정답
#         0.0001 (너무 낮음)      : 거의 모든 곳이 코너로 표시됨
#         0.6    (너무 높음)      : 가장 강한 몇 곳만 남음
def _harris(frame, ratio):
    g32 = np.float32(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    dst = cv2.dilate(cv2.cornerHarris(g32, 2, 3, 0.04), None)
    out = frame.copy()
    if dst.max() > 0:
        out[dst > ratio * dst.max()] = (0, 0, 255)
    return out


# Harris 임계 비율 보기 3종:  0.01 = 정답(적당)  /  0.0001 = 너무 낮아 거의 전부 코너  /  0.6 = 너무 높아 거의 안 남음
def s_harris_ok(f): return _harris(f, 0.01)
def s_harris_low(f): return _harris(f, 0.0001)
def s_harris_high(f): return _harris(f, 0.6)


# 3채널 BGR 에 cornerHarris 를 바로 적용하는 함정. 1채널이어야 하므로 OpenCV 에러가 화면에 뜬다 (흑백 변환을 빼먹은 실수)
def s_harris_color(f):
    return cv2.cornerHarris(f, 2, 3, 0.04)      # 3채널 입력 -> OpenCV 에러(1채널이어야 함)


# 무게중심 표시: 빨간 십자(+ 점)와 좌표 글자 "(cx, cy) = (..., ...)".
# 화면 밖 좌표(ix, iy 가 영상 범위를 벗어남)면 그리지 않고 'OFF SCREEN' 문구를 덧붙인다.
def _draw_blob_marker(img, c, p, label=True):
    ix, iy = _ci(p[0]), _ci(p[1])
    if 0 <= ix < VID_W and 0 <= iy < VID_H:
        cv2.drawMarker(img, (ix, iy), (0, 0, 255), cv2.MARKER_CROSS, 28, 3)
        cv2.circle(img, (ix, iy), 6, (0, 0, 255), -1)
    if label:
        outlined(img, "(cx, cy) = (%d, %d)%s" % (_ci(p[0]), _ci(p[1]),
                                                 "" if (0 <= ix < VID_W and 0 <= iy < VID_H) else "  OFF SCREEN"),
                 (10, 70), 0.5, C_WARN, 1)


# 미션 1의 마지막 단계(윤곽선 + 중심 좌표)에서 쓰는 보기 함수들. 형식은 모두 (st, ctx) -> st.
#   c_largest  : 정답 - 가장 큰 윤곽 + 모멘트로 중심
#   c_first    : 오답 - 목록의 첫 번째 윤곽 (점 노이즈일 수도 있음)
#   c_all      : 오답 - 전부 그림 (점 하나하나까지)
#   c_on_frame : 오답 - 마스크가 아닌 컬러 영상에 findContours (에러)
#   c_bbox     : 오답 - 모든 흰 픽셀을 감싸는 사각형
# ---- 미션 1: 윤곽선 파이프라인 ----
# 윤곽선 중 면적(contourArea)이 가장 큰 것을 초록으로 그리고, 모멘트로 무게중심을 구해 표시한다.
# 모멘트 M: m00 = 면적(픽셀 수),  m10 = x 좌표의 합,  m01 = y 좌표의 합
#   무게중심  cx = m10 / m00,  cy = m01 / m00
# m00 == 0 이면 나눗셈이 안 되므로 조건을 검사한다.
def c_largest(st, ctx):
    img = st["frame"].copy()
    cs = _contours(st["mask"])
    if cs:
        c = max(cs, key=cv2.contourArea)
        cv2.drawContours(img, [c], -1, (0, 255, 0), 3)
        # contour 의 공간 모멘트 dict (m00, m10, m01, ...)
        M = cv2.moments(c)
        if M["m00"] > 0:
            _draw_blob_marker(img, c, (M["m10"] / M["m00"], M["m01"] / M["m00"]))
    else:
        outlined(img, "no contour - show your face/hand", (10, 70), 0.5, C_WARN, 1)
    return S(st, vis=img)


# 윤곽선 목록의 첫 번째(cs[0])를 사용. 순서는 '큰 순서'가 아니므로 작은 노이즈 덩어리를 고를 수도 있다 -> 틀린 접근
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


# 모든 윤곽선을 그린다. 노이즈 점도 전부 포함되어 화면이 지저분하고 '하나의 대상'을 특정할 수 없다
def c_all(st, ctx):
    img = st["frame"].copy()
    cv2.drawContours(img, _contours(st["mask"]), -1, (0, 255, 0), 2)
    return S(st, vis=img)


# 컬러 영상(3채널)에 findContours 를 호출 -> 단일 채널이 아니라서 에러. 에러가 그대로 화면에 표시되는 '함정' 보기
def c_on_frame(st, ctx):
    _contours(st["frame"])      # 3채널 영상에 findContours -> OpenCV 에러가 그대로 화면에 뜬다
    return st


# cv2.boundingRect(mask): 마스크의 0이 아닌 모든 픽셀을 감싸는 최소 사각형. 흩어진 노이즈 점까지 포함해서 사각형이
#   비정상적으로 커질 수 있다
def c_bbox(st, ctx):
    img = st["frame"].copy()
    x, y, w, h = cv2.boundingRect(st["mask"])
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return S(st, vis=img)


# 미션 4 (Kalman 필터) 에서 쓰는 함수들이 시작되는 부분
# ---- 미션 4: Kalman ----
# Kalman 미션 1단계: "추적할 대상의 위치(측정값)를 어떻게 구하나?"
#   1) _skin_blob 으로 피부색 덩어리(윤곽) 하나를 찾는다.
#   2) variant(c) 함수로 그 윤곽의 '대표 좌표' p = (x, y) 를 계산한다.   <- 보기마다 이 계산 방법이 다르다
#   3) 화면에 윤곽(초록)과 좌표 표시(빨강 십자)를 그리고, st["meas"] = p 로 다음 단계(Kalman)에 넘긴다.
# 대상이 없으면(None) meas=None -> Kalman 쪽에서는 '측정 실패(가려짐)'로 처리된다.
def make_centroid(variant):
    # 피부색 덩어리를 찾아 variant 로 좌표를 구하고, 화면 표시와 측정값(meas)을 함께 돌려준다
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


# 정답: 무게중심 = (m10 / m00, m01 / m00).  m00 = 면적, m10 = x 좌표의 합, m01 = y 좌표의 합
def _cen_ok(c):
    M = cv2.moments(c)
    return M["m10"] / M["m00"], M["m01"] / M["m00"]


# 오답: x 와 y 를 바꿔 씀 -> 십자가 대각선 방향으로 엉뚱한 곳에 찍힌다
def _cen_swap(c):
    M = cv2.moments(c)
    return M["m01"] / M["m00"], M["m10"] / M["m00"]


# 오답: m00(면적)으로 나누는 것을 빼먹음 -> 수만~수백만 단위의 거대한 값이라 화면 밖(OFF SCREEN)
def _cen_raw(c):
    M = cv2.moments(c)
    return M["m10"], M["m01"]


# 오답: boundingRect 의 왼쪽 위 모서리 (x, y). 덩어리의 '중심'이 아니라 모서리라서 항상 한쪽으로 치우친다
def _cen_corner(c):
    x, y, w, h = cv2.boundingRect(c)
    return x, y


# 오답: 윤곽선의 첫 번째 점 c[0][0] = 윤곽선 위의 임의의 한 점.
#   중심이 아니다 (윤곽 배열은 (N, 1, 2) 모양이라 [0][0] 으로 접근)
def _cen_first(c):
    return float(c[0][0][0]), float(c[0][0][1])


# Kalman 상태 전이 행렬 F 세 가지.  상태 벡터 x = [x좌표, y좌표, x속도, y속도]ᵀ,  한 프레임이 지날 때 x' = F · x
#   KF_CV  (정답, 등속도 모델):  x' = x + vx,  y' = y + vy,  vx' = vx,  vy' = vy
#         -> "속도로 계속 움직인다"고 가정하므로 다음 위치를 '예측'할 수 있다
#   KF_NEG (오답): 속도 부호가 반대(x' = x - vx). 예측이 항상 진행 방향의 '반대쪽'으로 나간다
#   KF_ID  (오답): 단위행렬. 속도가 위치에 반영되지 않아 예측이 항상 '직전 위치' -> 측정값을 뒤늦게 따라감(지연)
KF_CV = [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]       # 등속 모델: 위치 += 속도
KF_NEG = [[1, 0, -1, 0], [0, 1, 0, -1], [0, 0, 1, 0], [0, 0, 0, 1]]    # 일부러 틀린 부호
KF_ID = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]       # 속도가 위치에 반영 안 됨
# 화살표(속도 방향)를 몇 프레임 앞까지 늘려서 그릴지: 화살표 끝 = 추정 위치 + 속도 x 8
KF_AHEAD = 8
# 가려짐 시뮬레이션: 3초 주기마다 마지막 1.2초 동안 측정값을 강제로 None 으로 만든다 (물체가 손에 가려진 상황 흉내)
LOSS_PERIOD, LOSS_LEN = 3.0, 1.2     # 3초마다 1.2초 동안 측정값이 사라진다(가려짐 흉내)


# cv2.KalmanFilter 객체를 만든다.  cfg 는 보기마다 다른 설정 dict (trans = 전이 행렬, r = 측정 잡음, dim = 상태 차원).
#   cv2.KalmanFilter(dynamParams=4, measureParams=2)  =  상태 4개(x, y, vx, vy), 측정 2개(x, y)
#
#   transitionMatrix F        : 시간이 지나면 상태가 어떻게 변하는가 (위 KF_* 참고)
#   measurementMatrix H       : 상태에서 '측정할 수 있는 부분'을 뽑는 행렬.
#                               [[1,0,0,0],[0,1,0,0]] = x, y 만 측정한다 (속도는 직접 못 잼)
#   processNoiseCov Q         : "모델이 얼마나 부정확한가". 작을수록 모델(등속도)을 믿어서 부드럽지만
#                               급한 방향 변화에 늦게 반응한다.
#                               속도 성분(5e-2)을 위치 성분(1e-2)보다 크게 두어 속도는 좀 더 유연하게 바뀌게 했다
#   measurementNoiseCov R     : "측정값이 얼마나 흔들리나". 작을수록 측정을 믿고, 클수록 모델 예측을 믿는다
#                               (아주 크면 거의 안 움직임)
#   errorCovPost P            : 초기 불확실성. 처음에는 아무것도 모르므로 크게(10) 잡아서 첫 측정에 빨리 맞춰지게 한다
#   statePost                 : 초기 상태 = 첫 측정 위치, 속도 0
#   statePre                  : predict() 이전의 사전 상태.
#                               statePost 와 같게 맞춰 놓지 않으면 첫 predict 가 이상한 값에서 출발한다
def _make_kf(cfg, x, y):
    if cfg.get("dim", 4) == 4:
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array(cfg.get("trans", KF_CV), np.float32)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.processNoiseCov = np.diag([1e-2, 1e-2, 5e-2, 5e-2]).astype(np.float32)
        kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0
        kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
    # dim=2 인 오답 모델: 상태가 (x, y) 뿐이고 전이 행렬이 단위행렬 -> '속도' 개념이 없어서 움직임을 예측하지 못하고 측정을
    #   뒤따라가기만 한다
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


# Kalman 단계의 핵심 함수. 매 프레임 다음 순서로 동작한다.
#   입력: st["meas"] = 이번 프레임에서 측정한 대상 위치 (없으면 None)
#         ctx = 이 보기 전용 저장소 (kf 객체, 마지막 측정, 궤적 등을 프레임 사이에 유지)
#         cfg = 모델 설정, lost_mode = 측정이 없을 때의 처리 방식, simulate_loss = 가려짐 시뮬레이션 여부
#
#   1) (선택) 가려짐 시뮬레이션: 주기 중 마지막 구간이면 측정을 None 으로 바꾸고 화면을 어둡게 한다
#   2) Kalman 필터가 아직 없고 측정이 들어왔으면 그때 생성한다 (첫 측정 위치에서 시작)
#   3) predict()   : 모델로 '다음 상태'를 예측                 (항상 호출)
#      correct(z)  : 측정 z 로 예측을 보정                      (측정이 있을 때만)
#      측정이 없으면 lost_mode 에 따라 처리:
#         predict : 보정 없이 예측값을 그대로 사용  <- 정답 (등속도로 계속 움직인다고 추정)
#         hold    : 마지막 측정값을 새 측정처럼 넣음 -> 화면이 멈춘 것처럼 보임
#         zeros   : (0, 0) 을 측정으로 넣음 -> 추정이 화면 왼쪽 위 구석으로 끌려감
#         origin  : 추정을 (0, 0) 으로 강제 -> 구석으로 순간이동
#         reset   : 필터를 버림 -> 다시 잡을 때까지 추정이 사라짐
#   4) 그리기: 빨강 원 = 측정, 초록 점 = 추정(필터 출력),
#     청록 선 = 최근 궤적(40점), 노랑 화살표 = 속도 x KF_AHEAD(앞으로의 위치)
def kalman_frame(st, ctx, cfg, lost_mode="predict", simulate_loss=False):
    """측정(meas) -> Kalman -> 화면. 빨강=측정, 초록=추정, 노랑 화살표=KF_AHEAD 프레임 뒤 예측."""
    img = st["frame"].copy()
    # ctx["now"]: Quiz 가 넣어 주는 현재 시각(초). 가려짐 주기 계산에 사용
    now = ctx.get("now", 0.0)
    # 이 보기가 처음 실행된 시각. 보기를 눌러 본 순간부터 주기가 시작되도록 기준을 저장
    t0 = ctx.setdefault("t0", now)
    meas = st.get("meas")
    # (경과시간 % 주기) 가 (주기 - 가려짐 길이) 이상이면 가려진 구간. 예) 주기 3.0, 길이 1.2 -> 1.8~3.0초 구간
    lost = simulate_loss and ((now - t0) % LOSS_PERIOD) >= (LOSS_PERIOD - LOSS_LEN)
    if lost:
        meas = None
        # 화면을 45% 밝기로 어둡게 -> '대상이 가려졌다'는 느낌을 준다 (float 로 곱한 뒤 uint8 로 복귀)
        img = (img.astype(np.float32) * 0.45).astype(np.uint8)
        outlined(img, "TARGET LOST (simulated occlusion)", (20, 200), 0.7, C_BAD, 2)
    # ctx 에 저장된 Kalman 필터. 첫 프레임이나 reset 후에는 None
    kf = ctx.get("kf")
    if kf is None and meas is not None:
        kf = ctx["kf"] = _make_kf(cfg, meas[0], meas[1])
        ctx["last"] = meas
        ctx["miss"] = 0
        ctx["trail"] = deque(maxlen=40)
    est = None
    if kf is not None:
        # predict(): x_pred = F · x_post  (모델로 한 걸음 앞 예측). 반환값은 (4, 1) 사전 추정
        pred = kf.predict()
        if meas is not None:
            # correct(z): 예측과 측정을 칼만 이득(Kalman gain)으로 섞어
            #   최종 추정(사후 추정)을 만든다. z 는 (2, 1) 열벡터여야 한다
            est = kf.correct(np.array([[meas[0]], [meas[1]]], np.float32))
            ctx["last"] = meas
            ctx["miss"] = 0
        else:
            ctx["miss"] = ctx.get("miss", 0) + 1
            # 정답: 측정이 없으니 correct 를 호출하지 않고 predict() 결과(pred)를 그대로 추정으로 쓴다 -> 속도 정보로 계속
            #   전진
            if lost_mode == "predict":
                est = pred
            # 오답: 마지막으로 본 위치를 새 측정처럼 correct -> 추정이 그 위치로 끌려가 멈춘다
            elif lost_mode == "hold":
                est = kf.correct(np.array([[ctx["last"][0]], [ctx["last"][1]]], np.float32))
            # 오답: (0, 0) 을 측정으로 넣음 -> 추정이 화면 왼쪽 위로 끌려간다
            elif lost_mode == "zeros":
                est = kf.correct(np.zeros((2, 1), np.float32))
            # 오답: 추정값 자체를 0 벡터로 덮어씀 -> 왼쪽 위 구석으로 순간이동
            elif lost_mode == "origin":
                est = np.zeros_like(pred)
            # 오답: 필터를 버림 -> 속도 정보도 사라지고 다시 대상을 찾을 때까지 추정이 없다
            elif lost_mode == "reset":
                ctx["kf"] = None
            # 시뮬레이션이 아닌 '진짜로' 25프레임(약 1초) 넘게 대상이 안 보이면 추적을 종료한다 (영원히 예측만 하며 날아가는
            #   것 방지)
            if not lost and ctx["miss"] > 25:       # 진짜로 오래 안 보이면 추적 종료
                ctx["kf"] = None
                est = None
    if meas is not None:
        cv2.circle(img, (_ci(meas[0]), _ci(meas[1])), 9, (0, 0, 255), 2)
    if est is not None:
        # est 를 (4,) 일반 배열로 펴서 v[0], v[1] = 위치, v[2], v[3] = 속도로 꺼낸다 (2차원 모델이면 길이 2)
        v = np.asarray(est, np.float64).flatten()
        ex, ey = float(v[0]), float(v[1])
        vx, vy = (float(v[2]), float(v[3])) if len(v) >= 4 else (0.0, 0.0)
        # 궤적 큐: 최근 40개 추정 위치만 보관 (오래된 것은 자동 삭제)
        tr = ctx.setdefault("trail", deque(maxlen=40))
        tr.append((_ci(ex), _ci(ey)))
        if len(tr) > 1:
            # 궤적을 청록색(BGR 255,255,0) 선으로 연결. 세 번째 인자 False = 닫지 않은(open) 선
            cv2.polylines(img, [np.array(tr, np.int32)], False, (255, 255, 0), 2)
        cv2.circle(img, (_ci(ex), _ci(ey)), 7, (0, 255, 0), -1)
        # 노랑 화살표: 추정 위치에서 (속도 x KF_AHEAD) 만큼 앞을 가리킴 = 'KF_AHEAD 프레임 뒤에 있을 위치'. 속도가 0이면
        #   화살표가 안 보인다
        cv2.arrowedLine(img, (_ci(ex), _ci(ey)), (_ci(ex + vx * KF_AHEAD), _ci(ey + vy * KF_AHEAD)),
                        (0, 255, 255), 3, tipLength=0.25)
    elif meas is None and not lost:
        outlined(img, "NO TARGET - show your face or hand", (10, 90), 0.5, C_WARN, 1)
    outlined(img, "red=measure  green=estimate  yellow=%d frames ahead" % KF_AHEAD, (10, 70), 0.45, C_TEXT, 1)
    return S(st, vis=img)


# kalman_frame 에 보기별 설정을 미리 묶어서 (st, ctx) -> st 형태의 단계 함수로 만든다 (클로저)
def make_kalman(cfg, lost_mode="predict", simulate_loss=False):
    # kalman_frame 에 미리 정한 설정(cfg, lost_mode, simulate_loss)을 넘겨 호출한다
    def run(st, ctx):
        return kalman_frame(st, ctx, cfg, lost_mode, simulate_loss)
    return run


# ---- 미션 5: MediaPipe 손 ----
# 미션 5 (MediaPipe 손) 단계 함수들은 모듈 수준 함수라서 Quiz 객체에 직접 접근할 수 없다.
# 그래서 이 dict 를 '공유 칸'으로 쓴다: Quiz 가 시작할 때 HandTracker 를 여기에 넣어 두면 h_detect 가 꺼내 쓴다.
_HANDS = {"tracker": None}      # Quiz 가 HandTracker 를 넣어 준다


# 손 스켈레톤 그리기: 연결선(초록) + 21개 관절 점. 손가락 끝(4, 8, 12, 16, 20)은 크고 빨갛게, 나머지는 작고 파랗게(BGR)
def draw_hand(img, pts):
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], (0, 255, 0), 2)
    for i, pt in enumerate(pts):
        tip = i in (4, 8, 12, 16, 20)
        cv2.circle(img, pt, 5 if tip else 3, (0, 0, 255) if tip else (255, 0, 0), -1)


# 미션 5 1단계(정답): MediaPipe 로 손을 검출한다. 실제 검출은 HandTracker.detect() 가 한다.
# 결과 lms = 손마다 21개의 랜드마크(정규화 좌표 x, y in 0~1, z)의 목록. st["hand"] 에 저장해서 다음 단계로 넘긴다.
def h_detect(st, ctx):
    t = _HANDS["tracker"]
    lms = t.detect(st["frame"]) if t is not None else []
    img = st["frame"].copy()
    if lms:
        outlined(img, "MediaPipe: %d hand(s), 21 landmarks each (x, y in 0~1)" % len(lms), (10, 70), 0.5, C_GOOD, 1)
    else:
        outlined(img, "No hand detected - show your hand to the camera", (10, 70), 0.5, C_WARN, 1)
    return S(st, hand=lms, vis=img)


# 오답 보기용: '모델에 이런 영상이 들어가게 된다'는 것만 보여 주고 검출은 하지
#   않는다 (hand=[]). BGR 을 그대로 넣거나 흑백을 넣는 실수를 시각적으로 확인
def h_model_view(fn, note):
    """틀린 입력 방식: 모델이 받게 될 영상을 그대로 보여 준다(검출은 하지 않는다)."""
    # 모델에 들어갈 영상(fn 적용 결과)과 설명 문구를 보여 주고, hand=[] 로 검출 없음 상태를 다음 단계에 넘긴다
    def run(st, ctx):
        v = to_view(fn(st["frame"])).copy()
        outlined(v, note, (10, 70), 0.5, C_WARN, 1)
        return S(st, vis=v, hand=[])
    return run


# 랜드마크 좌표 변환 보기들. MediaPipe 의 p.x, p.y 는 "이미지 폭/높이에 대한 비율(0~1)" 이다 (정규화 좌표).
#   정답 _pix_ok      : 픽셀 x = p.x * 폭(w),  픽셀 y = p.y * 높이(h)
#   오답 _pix_noscale : 곱하지 않음 -> 모든 점이 (0, 0) 근처 왼쪽 위 구석에 뭉침
#   오답 _pix_hw      : 폭과 높이를 바꿔 곱함 (x * h, y * w) -> 640x480 이므로 모양이 찌그러짐
#   오답 _pix_swap    : x 와 y 를 서로 바꿈 -> 손이 대각선으로 뒤집힌 위치
#   오답 _pix_flip    : y 를 (1 - y) 로 뒤집음 -> 손 모양이 위아래로 거꾸로 그려짐
def _pix_ok(p, w, h): return int(p.x * w), int(p.y * h)
def _pix_noscale(p, w, h): return int(p.x), int(p.y)  # 곱하기 누락 -> 전부 (0, 0) 근처
def _pix_hw(p, w, h): return int(p.x * h), int(p.y * w)  # 폭/높이 바꿔 곱함
def _pix_swap(p, w, h): return int(p.y * w), int(p.x * h)  # x, y 서로 바뀜
def _pix_flip(p, w, h): return int(p.x * w), int((1 - p.y) * h)  # y 뒤집힘


# 좌표 변환 방식 variant 를 적용해서 21개 랜드마크를 픽셀 좌표로 바꾸고 스켈레톤을 그린다. 결과 pts 를 다음 단계(손가락
#   판정)에 넘긴다
def make_pix(variant):
    # st['hand'] 의 각 손(21개 랜드마크)에 variant 변환을 적용해 스켈레톤을 그리고, 픽셀 좌표 목록을 st['pts'] 에 저장한다
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


# 손가락 판정 단계. 손가락마다 관절 3점(FINGER_ANGLE_POINTS)의 각도를 구하고 pred(각도) 가 참이면 '펴짐'으로 센다.
#   calculate_angle(a, b, c) : b 를 꼭짓점으로 하는 각 abc (도 단위). 펴지면 ~180, 접으면 작아짐
#   pred 가 보기마다 다른 규칙: 정답은 각도 >= 160. (예: 각도 <= 160 은 부등호 방향이 반대라 접힌 손가락을 센다)
#   화면: 손가락 끝에 초록 원(펴짐) / 빨강 원(접힘), 손목 근처에 개수 'Fingers: N'
# 계산은 정규화 좌표 lm(원본)으로 하고, 그리기는 픽셀 좌표 pts 로 한다.
def make_rule(pred):
    # 손마다 다섯 손가락의 관절 각도를 구해 pred 로 펴짐 여부를 판정하고, 손가락 끝 원 색과 개수를 그린다
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


# ---- 미션 정의 ----
# 미션 보기 튜플 생성 도우미: (코드, 단계함수, 정답여부, 해설ko, 해설en, 표시방식 'st'). 'st' 는 결과를 상태(st)에서 꺼내
#   그린다는 뜻 (기초 문제의 'image'/'mask' 와 구분)
def ch(code, fn, ok, ko, en):
    return (code, fn, ok, ko, en, "st")


# 5개의 미션을 만들어 돌려준다. 구조:
#
#   mission = dict(
#       id, name_ko, name_en,
#       needs_hands : MediaPipe 손 모델이 필요한 미션인가 (미션5)
#       source      : raw 프레임 -> 초기 상태 st  (문제 상황을 만드는 함수. 예: 뒤집고 노이즈를 섞음)
#       stages      : 단계 dict 의 리스트
#       final_lines : (미션 끝에서 만들어짐) 정답 보기 코드를 이어 붙인 '완성 코드' 줄들
#   )
#   stage = dict(
#       short_ko / short_en : 패널 위쪽 진행 표시에 쓰는 짧은 이름
#       unit_name           : 영어 단원 이름
#       ko / en             : 이 단계의 문제 문장
#       setup               : 이미 실행된 코드 줄들 (문자열 또는 (한글, 영어) 튜플)
#       metric              : 화면 숫자 지표 이름 또는 None
#       choices             : ch(...) 보기 5개
#   )
#
# ■ 파이프라인 동작 방식  (Quiz._stage_input 참고)
#    단계 k 의 입력 = source(raw) 에서 시작해서, 앞 단계 0..k-1 의 "정답 함수"를 차례로 모두 적용한 결과.
#    이 계산을 매 프레임 처음부터 다시 하기 때문에 카메라 영상이 움직여도 파이프라인 전체가 실시간으로 갱신된다.
#    (오답을 골라도 그 결과는 '화면에만' 보이고 다음 단계 입력에는 정답만 쓰인다 -> 재도전이 항상 같은 조건에서 가능)
def build_missions():
    # 샤프닝 커널 설명 줄 (기초 퀴즈와 같음)
    # 주의: filter2D 의 커널은 반드시 numpy 배열(float32)이어야 한다. 파이썬 리스트를 그대로 넣으면 OpenCV 가 에러를 낸다
    sharp_setup = "# sharp_kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]], np.float32)"
    # 미션 dict 들을 담을 리스트 (미션 1~5 순서)
    ms = []

    # ------------------------------------------------------------------
    # [미션 1]  뒤집힘 + 노이즈 -> 윤곽선        (단원: 반전 -> 필터링 -> 엣지)
    #   source : 원본을 상하 반전한 뒤 노이즈를 얹는다.  p_noise(p_flip_ud(raw))  (함수 안쪽부터 실행)
    #   1단계 뒤집기      cv2.flip(frame, 0)
    #   2단계 노이즈 제거  GaussianBlur 9x9      <- Canny 전에 꼭 필요한 전처리라는 흐름
    #   3단계 윤곽선       Canny(gray, 100, 200)
    ms.append(dict(
        id=1, name_ko="뒤집힘 + 노이즈 → 윤곽선", name_en="Flip + noise -> edges", needs_hands=False,
        source=lambda raw: st_new(raw, p_noise(p_flip_ud(raw))),
        stages=[
            # ---- 미션1 / 1단계: 상하 반전 되돌리기. 정답 결과(frame)가 2단계의 입력이 된다 ----
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
            # ---- 미션1 / 2단계: 노이즈 제거. 오답 중 (1, 1) 커널은 '커널이 너무 작으면 효과가 없다'는 것을 보여 주는 함정
            #   ----
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
            # ---- 미션1 / 3단계: 윤곽선 추출. f_view 를 쓰므로 결과는 화면에만 보이고 작업 영상은 그대로다 ----
            # 오답 설계: 임계값이 너무 낮음(10, 30) / 너무 높음(240, 255) / float32 에러 / 블러.
            #   1~2단계를 잘 풀어야 엣지가 깔끔하게 나오는 것을 확인할 수 있다.
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

    # ------------------------------------------------------------------
    # [미션 2]  색 바로잡기 -> 피부색 마스크 -> 점 제거 -> 윤곽/중심      (단원: 색공간 -> HSV -> 모폴로지 ->
    #   contour/moments)
    #   source : p_swap_rb(raw)  (R/B 채널이 바뀌어서 피부가 파랗게 보인다)
    #   1단계 BGR2RGB 로 색 복구   <- 색을 먼저 고쳐야 2단계의 HSV 피부색 범위가 맞는다
    #   2단계 inRange(hsv, ...)    f_mask 가 '카메라 노이즈로 생긴 흰 점'을 마스크에 섞어 준다
    #   3단계 MORPH_OPEN           점 제거
    #   4단계 findContours + 가장 큰 윤곽 + moments 로 중심
    ms.append(dict(
        id=2, name_ko="색 바로잡기 → 피부색 → 윤곽", name_en="Color -> skin -> contour", needs_hands=False,
        source=lambda raw: st_new(raw, p_swap_rb(raw)),
        stages=[
            # ---- 미션2 / 1단계 ----
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
            # ---- 미션2 / 2단계: 정답은 HSV 범위 inRange. 결과는 frame 이 아니라 mask 로 저장된다 (f_mask) ----
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
            # ---- 미션2 / 3단계: mask 에 모폴로지 적용 (f_maskop). 침식(erode)만 4번 하는 오답은 큰 덩어리까지 쪼그라드는
            #   것을 보여 준다 ----
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
            # ---- 미션2 / 4단계: mask -> contour -> 가장 큰 윤곽의 무게중심.
            #   이 보기들은 (st, ctx) 형태의 c_* 함수를 직접 사용 ----
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

    # ------------------------------------------------------------------
    # [미션 3]  어둡고 흐린 화면 -> CLAHE -> 샤프닝 -> Harris 코너       (단원: 영상 개선 -> 필터링 -> 코너 검출)
    #   source : p_dark(p_blur(raw))   (먼저 흐리게, 그다음 어둡게)
    #   1단계 LAB-L 에 CLAHE   (metric = 대비)   2단계 샤프닝 (metric = 선명도)   3단계 Harris 코너
    ms.append(dict(
        id=3, name_ko="어두운 화면 → 선명 → 코너", name_en="Dark -> sharp -> corners", needs_hands=False,
        source=lambda raw: st_new(raw, p_dark(p_blur(raw))),
        stages=[
            # ---- 미션3 / 1단계: 대비 복원. 람다로 clahe_on_L 결과를 새 frame 으로 저장 ----
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
            # ---- 미션3 / 2단계: 샤프닝. 1단계에서 밝아진 영상이 입력이다 ----
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
            # ---- 미션3 / 3단계: Harris 코너 ----
            # 핵심은 임계값(dst.max() 의 몇 배를 코너로 볼 것인가). 0.01 이 적당, 0.0001 은 너무 많이, 0.6 은 너무 적게
            #   표시된다.
            # cornerHarris 는 1채널 입력이 필수라서 컬러를 그대로 넣는
            #   보기는 에러가 난다. (setup 에 gray, dst 가 이미 있다고 가정)
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

    # ------------------------------------------------------------------
    # [미션 4]  피부색 물체의 중심 좌표 -> Kalman 필터 -> 가려짐 대응       (단원: contour/moments -> Kalman)
    #   source : st_new(raw, raw)   원본 그대로 시작 (이 미션은 '망가진 영상'이 아니라 '추적 문제')
    #   1단계 중심 좌표 구하기     st["meas"] 에 (cx, cy) 가 저장된다
    #   2단계 Kalman 모델 설정     등속도 모델의 전이 행렬을 고르는 문제  (예측 화살표가 움직임을 앞서야 정답)
    #   3단계 가려짐 처리          측정이 사라지는 순간(3초마다 1.2초) 어떻게 추정을 이어 갈 것인가
    # 각 단계의 입력은 앞 단계의 '정답' 결과이므로, 2단계에서는 정답인 무게중심 좌표가 필터에 들어간다.
    ms.append(dict(
        id=4, name_ko="움직임 추적 + Kalman 예측", name_en="Track + Kalman predict", needs_hands=False,
        source=lambda raw: st_new(raw, raw),
        stages=[
            # ---- 미션4 / 1단계: 측정값 만들기. 오답은 좌표가 엉뚱한 곳에 찍히거나(swap, corner, first) 화면 밖으로
            #   나가는(raw) 모습으로 확인 ----
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
            # ---- 미션4 / 2단계: 상태 전이 행렬 / 모델 선택 ----
            # 정답 KF_CV(등속도) 외 오답: KF_ID(속도 무시) / 2차원 모델(속도 없음) /
            #   KF_NEG(속도 부호 반대) / 측정 잡음 R 을 1e5 로 키움(측정 무시).
            # 물체를 좌우로 움직였을 때 노랑 화살표가 '진행 방향 앞'을 가리키는지 보면 차이를 알 수 있다.
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
            # ---- 미션4 / 3단계: 측정이 없을 때의 처리 ----
            # make_kalman(cfg, lost_mode, True) 의 세 번째 인자 True = 가려짐
            #   시뮬레이션 켜기 (주기적으로 측정 제거 + 화면 어두워짐).
            # 정답 'predict': correct 를 부르지 않고 predict() 결과만 사용 -> 가려진 동안에도 초록 점이 진행 방향으로 계속
            #   간다.
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

    # ------------------------------------------------------------------
    # [미션 5]  MediaPipe 손 인식 파이프라인       (단원: 색공간 RGB -> 정규화 좌표 -> 각도로 손가락 판정)
    #   needs_hands=True : 손 모델(.task) 파일이 없으면 메뉴에서 이 미션을 선택할 수 없다 (Quiz.mission_ok / hands_ok 참고)
    #   1단계 모델 입력 준비   BGR -> RGB 변환이 맞다 (MediaPipe 는 R, G, B 순서를 기대)
    #   2단계 좌표 변환        정규화 좌표(0~1) x 이미지 폭/높이 = 픽셀 좌표
    #   3단계 손가락 판정      관절 각도 >= 160 이면 '펴짐'
    # 이 미션의 실제 검출 결과는 st["hand"] -> st["pts"] -> 최종 손가락 수 순으로 단계마다 쌓인다.
    ms.append(dict(
        id=5, name_ko="MediaPipe 손 인식 파이프라인", name_en="MediaPipe hand pipeline", needs_hands=True,
        source=lambda raw: st_new(raw, raw),
        stages=[
            # ---- 미션5 / 1단계: 정답 보기만 실제로 h_detect 를 실행하고, 오답은 '모델이 보게 될 잘못된 입력 영상'을 보여
            #   준다(h_model_view) ----
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
            # ---- 미션5 / 2단계: 좌표 변환 5가지 (정답 1 + 곱하기 누락 / 폭-높이 바꿈 / x-y 바꿈 / y 뒤집기) ----
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
            # ---- 미션5 / 3단계: 임계 조건. 부등호 방향(>=, <=)과 기준값(20, 160, 179)에 따라 개수가 어떻게 달라지는지 손을
            #   폈다 쥐며 확인 ----
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

    # 미션을 끝냈을 때 보여 줄 "완성 코드" (정답 보기를 이어 붙이고, 필요한 준비 줄을 앞에 넣는다)
    # "완성 코드" 만들기용 데이터.
    # 키 (미션 id, 단계 번호 k) -> 그 단계의 정답 코드 '앞에' 끼워 넣을 준비 줄들.
    # 단계 보기의 코드 문자열은 핵심 한 줄만 있어서(예: `edges = cv2.Canny(gray, 100, 200)`),
    # 그대로 이어 붙이면 `gray` 가 정의되지 않는다. 그래서 필요한 변수 준비 줄(gray = ..., hsv = ... 등)을 이 표로 보충한다.
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
    # 미션마다 후처리:
    #   stg["_correct_fn"] : 각 단계의 정답 보기 함수를 저장 -> Quiz._stage_input 이 '앞 단계 정답을 차례로 적용'할 때 사용
    #   m["final_lines"]   : [준비 줄 + 정답 코드] 를 단계 순서대로 이어 붙인 완성 코드 -> 미션 클리어 때 화면/터미널에 출력
    for m in ms:
        lines = []
        for k, stg in enumerate(m["stages"]):
            # c[2] 가 정답 여부(True). 단계마다 정답은 정확히 1개라고 가정하고 첫 번째를 취한다
            right = [c for c in stg["choices"] if c[2]][0]
            stg["_correct_fn"] = right[1]
            lines.extend(final_pre.get((m["id"], k), []))
            lines.append(right[0])
        m["final_lines"] = lines
    # 미션 5개의 리스트 (각각 final_lines 와 단계별 _correct_fn 이 채워진 상태)
    return ms


# ======================================================================
# 5. 퀴즈 엔진 (메뉴 / 기초 문제 / 단계 미션 / 결과)
# ======================================================================
# [패널 좌표계]  아래의 버튼 사각형은 모두 "패널 안쪽 좌표"이다 (패널 왼쪽 위가 (0, 0)).
# 캔버스 전체 좌표에서 패널은 x = VID_W(640) 부터 시작하므로, 마우스 클릭 좌표에서 VID_W 를 빼서 비교한다 (Quiz.click).
# 사각형 표기는 (x0, y0, x1, y1) 이고 x0 <= x < x1, y0 <= y < y1 이 '안쪽'이다 (in_rect).
#
#   BTN_X = 12            버튼 왼쪽 여백
#   BTN_W = 460 - 24 = 436  양쪽에 12px 씩 여백을 둔 버튼 폭
BTN_X, BTN_W = 12, PANEL_W - 24
# 보기 버튼 안에서 코드 글씨가 차지하는 구간: 왼쪽 46px 은 번호 원(○), 오른쪽 8px 은 여백 -> 가로 382px.
#   fit_lines 가 이 폭에 코드를 맞춘다.
CODE_X = 46
CODE_W = BTN_W - CODE_X - 8
# 보기 코드 글자 배율 후보 (큰 것부터). 0.40 은 한 줄에 겨우 들어갈 때만 쓰고, 그래도 안 되면 0.43 으로 두 줄로 나눈다
CODE_SCALES = (0.43, 0.40)
SETUP_SCALES = (0.36, 0.34)          # setup(이미 실행된 코드) 줄 글자 배율 후보
CLEAR_SCALES = (0.40, 0.38, 0.36, 0.34)  # 미션 완성 코드 박스 글자 배율 후보
# 보기 버튼 배치 (첫 버튼 y, 버튼 높이, 버튼 사이 간격).
#   기초 문제   (172, 48, 4): 문제 문장 영역이 짧아서 172 부터 시작, 버튼 5개 -> y = 172, 224, 276, 328, 380 (끝 428)
#   단계 미션   (204, 41, 4): 위에 '단계 표시 + setup 코드 줄'이 더 필요해서 204 부터 시작, 버튼은 조금 낮게(41) -> 끝 425
GEO_BASIC = (172, 48, 4)       # (첫 버튼 y, 버튼 높이, 간격)
GEO_MISSION = (204, 41, 4)
# '다음 문제 ▶' 버튼 (정답을 맞힌 뒤에 나타남). 패널 오른쪽 아래, 폭 128 x 높이 32
NEXT_RECT = (PANEL_W - 140, 440, PANEL_W - 12, 472)
# 결과 요약 화면의 세 버튼: 처음부터(R) / 틀린 문제만(W) / 메뉴(M). 모두 패널 아래쪽 y 410~446 에 가로로 나란히 배치
SUM_RESTART = (12, 410, 148, 446)
SUM_WRONG = (156, 410, 302, 446)
SUM_MENU = (310, 410, 448, 446)
# 메뉴 항목 배치: 첫 항목 y=44, 높이 44, 간격 4 -> 항목 8개가 y 44 ~ 428 에 들어간다
MENU_Y0, MENU_H, MENU_GAP = 44, 44, 4
# 메뉴 항목 8개의 종류 이름 (순서가 곧 키 1~8 번호).
#   basic = 기초 퀴즈 8문제, m1~m5 = 단계 미션 1~5, all = 전체 도전, free = 자유 실습 모드
MENU_KINDS = ["basic", "m1", "m2", "m3", "m4", "m5", "all", "free"]


# i 번째(0부터) 보기 버튼의 사각형. geo 로 기초/미션 배치를 선택. 위치 = 첫 y + i x (높이 + 간격)
def btn_rect(i, geo=GEO_BASIC):
    y0, h, gap = geo
    y = y0 + i * (h + gap)
    return (BTN_X, y, BTN_X + BTN_W, y + h)


# i 번째 메뉴 항목 사각형
def menu_rect(i):
    y = MENU_Y0 + i * (MENU_H + MENU_GAP)
    return (BTN_X, y, BTN_X + BTN_W, y + MENU_H)


# 점 (x, y) 가 사각형 r = (x0, y0, x1, y1) 안에 있는가. 오른쪽/아래쪽 끝은 포함하지 않는다(반열린 구간)
def in_rect(r, x, y):
    return r[0] <= x < r[2] and r[1] <= y < r[3]


# setup 줄은 문자열(한/영 공통 코드) 또는 (한글, 영어) 튜플이다. 현재 UI 언어(ko)에 맞는 쪽을 골라 준다
def _setup_text(line, ko):
    return line if isinstance(line, str) else (line[0] if ko else line[1])


# 퀴즈 전체를 담당하는 클래스: 상태 관리 + 한 프레임 그리기(render) + 입력 처리(key/click/move).
#
# ■ 상태(self.state) 흐름
#       menu ──(세트 선택)──> ask ──(보기 선택)──┬─ 오답: 그 결과를 보여 주고 ask 유지 (다시 선택 가능)
#         ^                    ^                  └─ 정답: solved ──(SPACE/다음)──> 다음 문제 ask ... 마지막 후 summary
#         └────────(M 키)──────┴──────────────────────────────────────────────────────────┘
#    menu    : 세트(기초 / 미션 1~5 / 전체 / 자유) 선택 화면
#    ask     : 문제가 떠 있고 보기를 고르는 중 (오답을 눌러도 계속 ask)
#    solved  : 정답을 맞힌 상태. 0.5초 페이드 + 해설 + '다음' 버튼
#    summary : 모든 문제를 끝낸 결과 화면 (점수, 틀린 문제, 미션의 완성 코드)
#
# ■ 점수: 문제마다 처음 고른 보기가 정답이면 10점, 한 번 틀리고 맞히면 5점, 건너뛰기(S)는 0점.
#
# ■ 화면 그리기를 가볍게 하는 장치
#    - self.base_cache / banner_cache / menu_cache / sum_cache : 글자(한글) 렌더링 결과를 캐시.
#      문제가 바뀔 때만 Pillow 로 새로 그리고, 매 프레임은 캐시된 알파 마스크를 붙이기만 한다.
#    - 단계 입력은 매 프레임 계산하지만 비용은 수 ms 수준이라 실시간(30FPS)을 유지한다.
class Quiz:
    # text  : TextRenderer (한글/영문 글씨 그리기)
    # seed  : 보기 순서를 섞을 난수 시드 (--seed). None 이면 실행마다 다름
    # hands : HandTracker 또는 None (미션5 / 자유 실습 손 인식용)
    # start : 'menu' 면 메뉴부터. 그 외('basic', 'm2', 'all' ...)면 메뉴를 건너뛰고 해당 세트 바로 시작 (--set)
    def __init__(self, text, seed=None, hands=None, start="menu"):
        self.text = text
        # 이 Quiz 전용 난수 생성기 (전역 random 과 분리해서 --seed 로 보기 순서를 재현할 수 있게 함)
        self.rng = random.Random(seed)
        # 기초 문제 8개의 원본 데이터
        self.bank = build_questions()
        # 미션 5개의 원본 데이터
        self.missions = build_missions()
        self.hands = hands
        # 미션5 단계 함수(h_detect)가 모듈 수준에서 손 인식기를 쓸 수 있도록 공유 칸에 등록
        _HANDS["tracker"] = hands
        # 문제/단계 상단(제목, 문제문장, setup 코드)의 글자 렌더링 캐시
        self.base_cache = {}
        # 영상 아래쪽 '정답!/오답 + 해설' 배너의 글자 렌더링 캐시
        self.banner_cache = {}
        # 메뉴 화면 글자 캐시
        self.menu_cache = {}
        # 결과 요약 화면 글자 캐시
        self.sum_cache = None
        # 마우스가 올라가 있는 보기 번호 (버튼 하이라이트용), 없으면 None
        self.hover = None
        # 현재 진행 중인 세트 종류 (MENU_KINDS 중 하나)
        self.kind = "all"
        # 메뉴에 잠깐 띄우는 안내 문구 (예: 손 모델이 없어 미션5를 열 수 없을 때)
        self.menu_msg = ""
        # 손 모델 사용 가능 여부의 캐시 (None = 아직 확인 안 함). 확인은 모델 로딩을 동반해서 한 번만 한다
        self._avail = None
        # 지금 세트의 문제 목록 (기초 문제 + 미션 단계를 같은 형태의 dict 로 펼친 리스트)
        self.questions = []
        # 현재 문제의 보기 버튼 배치 (기초 / 미션 중 하나)
        self.geo = GEO_BASIC
        self._reset_run()
        self.state = "menu"
        if start != "menu":
            self.start(start)

    # Quiz 의 상태 관리 메서드들: 초기화 / 문제 목록 만들기 / 보기 선택 / 다음 문제로 이동 등
    # ---- 상태 ----
    # 한 번의 '세트 진행'에 필요한 값들을 모두 처음 상태로 되돌린다 (세트를 새로 시작하거나 다시 도전할 때 호출)
    def _reset_run(self):
        # qi      : 현재 문제 번호 (self.questions 리스트의 인덱스)
        # score   : 누적 점수
        # streak  : 연속으로 '첫 시도에 정답'을 맞힌 횟수 (오답/건너뛰기에서 0으로 초기화)
        # results : {문제 id: dict(tries=시도 횟수, pts=얻은 점수,
        #   skipped=건너뜀 여부)} -> 결과 요약 화면과 '틀린 문제만 다시'에 사용
        self.qi = 0
        self.score = 0
        self.streak = 0
        self.results = {}
        # picked  : 지금 선택한 보기 번호 (아직 안 골랐으면 None)
        # wrong   : 이번 문제에서 이미 눌러 본 오답 보기 번호들의 집합 -> 같은 오답을 또 누르지 못하고, 버튼이 붉게 표시됨
        # tries   : 이번 문제에서 보기를 누른 횟수 (점수 계산: 1번째 정답 10점 / 2번째 5점 / 그 이상 0점)
        # t_solved: 정답을 맞힌 시각 (0.5초 페이드 계산 기준)
        self.picked = None
        self.wrong = set()
        self.tries = 0
        self.t_solved = None
        # pctx    : 파이프라인 앞 단계의 정답 함수들이 쓰는 저장소 {(미션 id, 단계 j): ctx}.  Kalman 처럼 프레임 사이에
        #   상태를 가지는 단계용
        # cctx    : 지금 문제의 '보기별' 저장소 {(문제 id, 보기 번호): ctx}.  보기를 새로 고르면 새로 시작 (pick 참고)
        # ema     : 숫자 지표를 부드럽게 만드는 이동평균 값 저장소 (_ema 참고)
        # cleared : 이미 '미션 클리어' 코드를 터미널에 출력한 미션 id 집합 (중복 출력 방지)
        self.pctx = {}
        self.cctx = {}
        self.ema = {}
        self.cleared = set()

    # MediaPipe 손 인식을 쓸 수 있는가?  (사용 가능 여부, 이유 문자열) 튜플을 돌려준다.
    # 처음 한 번만 HandTracker.available() 로 확인하고 self._avail 에 캐시한다 (모델 파일/패키지 존재 검사).
    def hands_ok(self):
        if self._avail is None:
            if self.hands is None:
                self._avail = (False, "no hand tracker")
            else:
                self._avail = self.hands.available()
        return self._avail

    # 미션 m 을 실행할 수 있는가: 손이 필요 없는 미션이면 항상 가능, 필요하면 hands_ok() 의 첫 값(사용 가능 여부)을 따른다
    def mission_ok(self, m):
        return (not m["needs_hands"]) or self.hands_ok()[0]

    # 세트 종류에 해당하는 미션 목록. 'missions'/'all' = 전부, 'm3' 같은 값 = 그 번호 미션 하나, 'basic' = 빈 리스트
    def _selected_missions(self, kind):
        if kind in ("missions", "all"):
            return list(self.missions)
        return [m for m in self.missions if "m%d" % m["id"] == kind]

    # 세트 종류에 맞는 "문제 목록"을 만든다. 기초 문제와 미션 단계를 같은 모양의 dict 로 펼쳐서 한 리스트에 담는다.
    #   기초 문제: id = 0~7,            group='basic',  mission=None
    #   미션 단계: id = 100 x 미션번호 + 단계번호 (예: 미션2의 3번째 단계 = 202),  group='m2',  mission=<미션 dict>
    #   id 를 고정 규칙으로 만드는 이유: 결과 기록(results), 보기별 ctx,
    #     배너 캐시의 키로 쓰고, '틀린 문제만 다시'에서 id 로 골라내기 위해서
    # dict(q) 로 복사하는 이유: 보기 순서를 섞을 때 원본 데이터(self.bank 등)가 바뀌지 않게 하려고
    def _build_list(self, kind):
        qs = []
        if kind in ("basic", "all"):
            for i, q in enumerate(self.bank):
                d = dict(q)
                # 원본에 id, 그룹, 미션 정보(없음)를 덧붙인다. stage_i 는 미션 안에서의 단계 번호 (기초 문제는 0)
                d.update(id=i, group="basic", mission=None, stage_i=0)
                qs.append(d)
        for m in self._selected_missions(kind):
            # 손 모델을 쓸 수 없는 상태면 손이 필요한 미션은 목록에서 건너뛴다 ('전체 도전'에서 미션5만 빠지는 이유)
            if not self.mission_ok(m):
                continue
            for k, stg in enumerate(m["stages"]):
                d = dict(stg)
                # 미션 단계는 problem=None: 망가뜨리기는 미션의 source() 가 대신 하므로 개별 단계에는 필요 없다
                d.update(id=100 * m["id"] + k, group="m%d" % m["id"], mission=m, stage_i=k,
                         unit="M%d" % m["id"], unit_name=m["name_en"], problem=None)
                qs.append(d)
        return qs

    # 메뉴에서 세트를 고르면 호출된다.  반환값: 'free' (자유 실습으로 전환하라고 App 에 알림) 또는 None.
    # 순서:
    #   1) 'free' 면 바로 'free' 를 돌려준다
    #   2) 미션5 단독 선택인데 손 인식을 못 쓰면 -> 메뉴에 이유를 띄우고 시작하지 않음 (menu_msg)
    #   3) 손이 필요한 미션이 포함되어 있으면 지금 모델을 불러온다 (ensure; 수 초 걸릴 수 있음). 실패하면 사용 불가로 기록
    #      ('전체 도전'에서는 실패해도 손 미션만 빠지고 계속 진행)
    #   4) self.kind 를 기록하고 세션을 처음부터 시작
    def start(self, kind):
        """메뉴에서 세트를 고른다. 'free' 면 자유 실습으로 넘기라고 App 에 알린다."""
        if kind == "free":
            return "free"
        sel = self._selected_missions(kind)
        if kind.startswith("m") and kind != "missions" and sel and not self.mission_ok(sel[0]):
            self.menu_msg = "Unavailable: " + self.hands_ok()[1]
            return None
        if any(m["needs_hands"] and self.mission_ok(m) for m in sel):
            # ensure(): 모델 파일을 찾아 MediaPipe 를 초기화. 처음 한 번만 오래 걸리고 이후엔 즉시 True
            if not self.hands.ensure():          # 모델을 지금 불러온다 (잠깐 멈출 수 있음)
                self._avail = (False, self.hands.err)
                if kind.startswith("m") and kind != "missions":
                    self.menu_msg = "MediaPipe unavailable: " + self.hands.err
                    return None
        self.kind = kind
        self.menu_msg = ""
        self.reset_session()
        return None

    # 현재 세트로 새 진행을 시작한다. subset 이 있으면 그 id 의 문제만 남긴다 ('틀린 문제만 다시' 용).
    # 시작할 때마다 각 문제의 보기 순서를 무작위로 섞는다 -> 정답이 항상 같은 번호에 오지 않고, 다시 풀 때도 위치를 외워서
    #   맞힐 수 없다.
    def reset_session(self, subset=None):
        qs = self._build_list(self.kind)
        if subset is not None:
            keep = set(subset)
            qs = [q for q in qs if q["id"] in keep]
        for q in qs:
            # 리스트를 복사해서 섞는다 (원본 튜플 순서는 유지)
            ch_ = list(q["choices"])
            # Quiz 전용 난수(seed 지정 가능)로 섞기
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

    # 새 문제로 들어갈 때 문제별 상태(선택, 오답 기록, 시도 횟수, ctx, 지표 평균)를 비우고, 미션 단계인지에 따라 보기 버튼
    #   배치(geo)를 정한다
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

    # 지금 풀고 있는 문제 dict
    def current(self):
        return self.questions[self.qi]

    # q 가 미션의 '마지막 단계'인가 (기초 문제는 mission 이 None 이라
    #   False). 마지막 단계를 맞히면 '미션 클리어'와 완성 코드 출력
    def is_last_stage(self, q):
        return q["mission"] is not None and q["stage_i"] == len(q["mission"]["stages"]) - 1

    # 보기 idx 를 선택한다 (마우스 클릭 또는 숫자 키).
    #   - 문제를 푸는 중(ask)이 아니거나, 범위 밖이거나, 이미 눌러 본 오답이면 무시
    #   - 정답이면 상태를 solved 로 바꾸고 점수 기록. (미션 마지막 단계면 완성 코드 출력)
    #   - 오답이면 wrong 에 기록만 하고 ask 를 유지 -> 다른 보기를 다시 고를 수 있다
    def pick(self, idx, now):
        if self.state != "ask" or idx is None:
            return
        q = self.current()
        if not (0 <= idx < len(q["choices"])) or idx in self.wrong:
            return
        ok = q["choices"][idx][2]
        self.tries += 1
        self.picked = idx
        # 보기를 새로 고르면 그 보기의 ctx(Kalman 등 프레임 간 상태)를 지워서 처음부터 다시 시작하게 한다
        self.cctx.pop((q["id"], idx), None)
        # '적용 후' 지표(n)의 이동평균을 비워서, 직전에 눌렀던 보기의 숫자가 새 보기 숫자와 섞이지 않게 한다
        self.ema.pop("n", None)
        if ok:
            self.state = "solved"
            self.t_solved = now
            # 점수 규칙: 첫 시도 10점, 두 번째 시도 5점, 세 번째 이후 0점
            pts = 10 if self.tries == 1 else (5 if self.tries == 2 else 0)
            self.score += pts
            # 연속 정답(streak): 첫 시도에 맞혔을 때만 증가, 두 번 이상 걸렸으면 0으로
            self.streak = self.streak + 1 if self.tries == 1 else 0
            self.results[q["id"]] = dict(tries=self.tries, pts=pts, skipped=False)
            if self.is_last_stage(q):
                self._announce_clear(q["mission"])
        else:
            self.wrong.add(idx)
            self.streak = 0

    # 미션을 클리어했을 때 터미널에 완성 코드를 출력한다 (미션당 한 번만). 화면 쪽 출력은 결과 요약 화면(_summary_panel)이
    #   맡는다
    def _announce_clear(self, m):
        if m["id"] in self.cleared:
            return
        self.cleared.add(m["id"])
        print("\n[MISSION %d CLEAR] %s -- final pipeline:" % (m["id"], m["name_en"]))
        for ln in m["final_lines"]:
            print("    " + ln)

    # 문제 건너뛰기 (S 키). 0점 처리하고 연속 정답을 끊는다. 풀이 중(ask)일 때만 가능
    def skip(self):
        if self.state != "ask":
            return
        q = self.current()
        self.results[q["id"]] = dict(tries=self.tries, pts=0, skipped=True)
        self.streak = 0
        self._advance()

    # 정답을 맞힌 상태(solved)에서만 다음 문제로 넘어간다 (SPACE / ENTER / '다음' 버튼)
    def next(self):
        if self.state == "solved":
            self._advance()

    # 문제 번호를 하나 올린다. 마지막이었으면 결과 요약(summary) 화면으로, 아니면 다음 문제를 준비
    def _advance(self):
        self.qi += 1
        if self.qi >= len(self.questions):
            self.state = "summary"
            self.sum_cache = None
        else:
            self.state = "ask"
            self._enter_question()

    # 결과 화면의 R: 같은 세트를 처음부터 다시 (보기 순서도 다시 섞임)
    def restart_all(self):
        self.reset_session()

    # 결과 화면의 W: 첫 시도에 못 맞힌(10점 미만) 문제만 모아 다시 풀기. 모두 만점이면 아무것도 하지 않는다
    def restart_wrong(self):
        ids = [q["id"] for q in self.questions
               if self.results.get(q["id"], {}).get("pts", 0) < 10]
        if ids:
            self.reset_session(ids)

    # 메뉴로 돌아가기 (M 키 / 결과 화면의 메뉴 버튼)
    def to_menu(self):
        self.state = "menu"
        self.hover = None
        self.menu_msg = ""

    # 입력 처리. key / click 은 "App 이 해야 할 동작"을 돌려준다:  'free' = 자유 실습 모드로 전환,  None = 아무것도 안 함.
    # (Quiz 는 자기 안의 상태만 바꾸고, 모드 전환 같은 큰 일은 App 이 한다)
    # ---- 입력 (App 에게 돌려줄 동작: "free" 또는 None) ----
    # 키보드 입력 (k = cv2.waitKey 가 돌려준 키 코드, now = 현재 시각).
    #   메뉴      : 1~8 -> 해당 세트 시작
    #   모든 화면 : M -> 메뉴
    #   결과 화면 : R 처음부터 / W 틀린 문제만
    #   문제 화면 : 1~9 보기 선택 / SPACE(32), ENTER(13) 다음 / S 건너뛰기
    def key(self, k, now):
        # ASCII 범위(0~127) 키만 문자로 바꾸고 소문자로 통일 (대소문자 구분 없이 처리). 방향키 같은 특수 키는 빈 문자열이
        #   되어 무시
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

    # 마우스 왼쪽 클릭 처리.  (x, y) 는 "논리 캔버스" 좌표 (App.mouse_cb 가 확대 배율을 이미 나눠서 보내 준다).
    # 영상 영역(x < 640) 클릭은 무시하고, 패널 영역이면 패널 안쪽 좌표 (px, py) = (x - 640, y) 로 바꿔서
    # 상태(메뉴 / 결과 / 문제)에 맞는 버튼 사각형과 비교한다.
    def click(self, x, y, now):
        if x < VID_W:
            return None
        # 패널 안쪽 좌표로 변환 (패널 왼쪽 위가 원점)
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

    # 마우스 이동 처리: 마우스가 올라간 버튼 번호를 self.hover 에 기록한다 ('next' 는 다음 버튼). 그리기에서 하이라이트에
    #   사용
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

    # 화면 그리기 메서드들
    # ---- 그리기 ----
    # 한 프레임의 전체 화면(논리 캔버스 1100x480)을 만든다.  raw = 카메라 원본(640x480 BGR),  now = 현재 시각.
    #   메뉴  : 왼쪽에 라이브 카메라 + 오른쪽에 메뉴 패널
    #   요약  : 왼쪽 카메라를 45% 밝기로 어둡게(결과에 집중) + 오른쪽에 요약 패널
    #   문제  : 왼쪽 = _video()가 만든 (문제/결과) 화면, 오른쪽 = _panel() 의 문제 패널, 왼쪽 위에 상태 태그 글자
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

    # 기초 문제 보기 결과를 화면용으로 후처리: 표시 방식이 'mask' 면 원본(base)에 마스크를 적용해 보여 주고, 'image' 면
    #   to_view 로 표준 형태만 맞춘다
    def _post(self, base, out, kind):
        if kind == "mask":
            return apply_mask(base, out)
        return to_view(out)

    # 미션 단계의 입력 상태 st 를 만든다 (파이프라인 핵심).
    #   1) st = 미션의 source(raw)  : 원본을 망가뜨린 초기 상태
    #   2) 이 단계 이전의 단계 0..(k-1) 의 "정답 함수"를 순서대로 적용
    #      -> 오답을 골라 본 결과는 다음 단계에 전달되지 않는다. 항상 정답으로 풀린 상태에서 다음 단계가 시작된다
    #   3) 각 앞 단계마다 self.pctx 의 전용 ctx 를 쓴다 (Kalman 처럼 프레임 사이의 상태 유지). ctx["now"] 로 현재 시각을 전달
    # 이 함수는 매 프레임 호출되므로 카메라 영상이 바뀌면 파이프라인 전체가 새 프레임으로 다시 계산된다.
    def _stage_input(self, q, raw, now):
        """미션 단계의 입력: 원본을 망가뜨린 뒤, 앞 단계의 정답 보기를 차례로 적용한 상태."""
        m = q["mission"]
        st = m["source"](raw)
        for j in range(q["stage_i"]):
            ctx = self.pctx.setdefault((m["id"], j), {})
            ctx["now"] = now
            st = m["stages"][j]["_correct_fn"](st, ctx)
        return st

    # 지수 이동평균: 새 값 = 0.75 x 이전 값 + 0.25 x 현재 값. 노이즈 때문에 프레임마다 흔들리는 지표 숫자를 읽기 쉽게
    #   부드럽게 만든다
    def _ema(self, key, v):
        old = self.ema.get(key)
        self.ema[key] = v if old is None else 0.75 * old + 0.25 * v
        return self.ema[key]

    # 왼쪽 영상 영역(640x480)을 만든다.  반환: (영상, 왼쪽 위 태그 문자열, 태그 색).
    #
    #   [1] base = 보기를 누르기 전에 보이는 '문제 화면'
    #         - 미션 단계: _stage_input() 결과 (앞 단계 정답까지 적용된 상태)
    #         - 기초 문제: problem 함수로 망가뜨린 영상 (broken)
    #   [2] 아직 아무것도 안 골랐으면 base 에 지표 숫자(있으면)만 얹어 반환
    #   [3] 보기를 골랐으면 그 보기의 함수를 실행해서 res(결과 화면)를 만든다
    #         - 미션 단계(kind == 'st'): fn(st_in, ctx) -> 상태 -> vis_img  (ctx = 이 보기 전용 저장소)
    #         - 기초 문제: fn(broken) -> _post() 로 표시 형태 보정
    #         - 보기가 예외를 내면(예: Canny 에 float32) 에러 메시지를 화면에 띄우고 base 를 보여준다 (오답의 학습 포인트)
    #   [4] 정답이면 base -> res 로 FADE_SEC(0.5초) 동안 크로스페이드 (addWeighted), 오답이면 res 를 그대로 보여 준다
    #   [5] 오른쪽 위에 작은 'BEFORE'(base) 화면을 얹어 전/후를 비교하게 하고, 지표 숫자와 해설 배너를 그린다
    def _video(self, q, raw, now):
        err = None
        # broken: 기초 문제에서 '망가뜨린 영상'. 보기 함수의 입력으로 쓰인다 (미션 단계에서는 사용 안 함)
        broken = None
        st_in = None
        if q["mission"] is not None:
            # 미션 단계의 입력 계산은 예외가 날 수 있다(예: 앞 단계가 MediaPipe 에러). 죽지 않고 원본 + ERROR 문구를 보여
            #   주도록 감싼다
            try:
                st_in = self._stage_input(q, raw, now)
                base = vis_img(st_in)
            except Exception as e:
                base = to_view(raw)
                # 에러 메시지는 핵심 이유만 한 줄로 줄여서 화면에 표시 (OpenCV 에러는 경로까지 붙어 매우 길다. short_err 참고)
                err = short_err(e)
        else:
            broken = q["problem"](raw) if q["problem"] else raw
            base = to_view(broken)
        if q["mission"] is not None:
            head = "STEP %d/%d" % (q["stage_i"] + 1, len(q["mission"]["stages"]))
        else:
            head = "PROBLEM VIEW"
        # 아직 보기를 안 골랐다 -> 문제 화면(base)만 보여 준다
        if self.picked is None:
            view = base.copy()
            self._metric(view, q, base, None)
            if err:
                outlined(view, clip_text("ERROR: " + err, VID_W - 20, 0.45), (10, VID_H - 12), 0.45, C_BAD, 1)
            return view, head + " - pick the fix", C_BAD
        # 선택한 보기 튜플 풀기: (코드 문자열, 함수, 정답 여부, 해설 ko, 해설 en, 표시 방식)
        code, fn, ok, why_ko, why_en, kind = q["choices"][self.picked]
        # 선택한 보기를 실제로 실행하는 부분
        try:
            # kind == 'st' 는 미션 단계 보기: 함수 시그니처가 (st, ctx) -> st
            if kind == "st":
                if st_in is None:
                    raise RuntimeError(err or "previous step failed")
                # 이 보기 전용 ctx. 같은 보기를 계속 보는 동안 유지되어 Kalman 필터 상태 등이 프레임 사이에 이어진다
                ctx = self.cctx.setdefault((q["id"], self.picked), {})
                ctx["now"] = now
                res = vis_img(fn(st_in, ctx))
            else:
                res = self._post(base, fn(broken), kind)
            err = None
        # 예외를 숨기지 않고 화면에 보여 준다: 'float32 를 Canny 에 넣으면 에러' 같은 함정을 직접 체험하게 하려는 의도
        except Exception as e:  # 오답 보기가 에러를 내는 것도 학습 포인트
            res = base
            err = short_err(e)
        # 정답을 맞힌 직후에만 페이드 효과를 쓴다 (오답은 결과를 그대로 즉시 표시)
        if ok and self.state == "solved":
            # 페이드 진행도 0.0 -> 1.0 (정답을 맞힌 순간부터 0.5초 동안)
            t = min(1.0, (now - self.t_solved) / FADE_SEC)
            # 크로스페이드: 결과 = base x (1 - t) + res x t.  t=0 이면 문제 화면, t=1 이면 정답 결과 화면
            view = cv2.addWeighted(base, 1.0 - t, res, t, 0)
            tag, col = ("MISSION CLEAR" if self.is_last_stage(q) else "SOLVED"), C_GOOD
        else:
            view = res.copy()
            tag, col = "YOUR PICK - wrong", C_WARN
        # 오른쪽 위에 200x150 크기의 'BEFORE'(보기를 적용하기 전 화면)를 붙인다. 위치: x 430~630, y 10~160
        # 전/후 비교용 작은 화면
        ins = cv2.resize(base, (200, 150), interpolation=cv2.INTER_AREA)
        # 슬라이싱 대입: 행 10~159, 열 430~629 (= 150행 x 200열) 영역에 축소 영상을 덮어쓴다
        view[10:160, VID_W - 210:VID_W - 10] = ins
        cv2.rectangle(view, (VID_W - 210, 10), (VID_W - 10, 160), (255, 255, 255), 1)
        outlined(view, "BEFORE", (VID_W - 204, 28), 0.5, C_TEXT, 1)
        self._metric(view, q, base, res, ok)
        self._banner(view, q, ok, err)
        return view, tag, col

    def _metric(self, view, q, base, res, ok=False):
        """왼쪽 위에 효과를 숫자로 보여 준다 (선명도 / 노이즈 / 대비).
        숫자가 좋아져도 오답(예: Canny)일 수 있어서, 초록색은 정답일 때만 쓴다."""
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

    # 영상 아래쪽에 깔리는 반투명 해설 배너: '정답!/오답' + 선택한 보기의 해설(why).
    # 글자(한글) 렌더링은 비싸므로 (문제 id, 고른 보기, 언어)를 키로 캐시하고, 매 프레임은 캐시를 붙이기만 한다.
    def _banner(self, view, q, ok, err):
        # 캐시 키. 같은 문제에서 다른 보기를 누르면 다른 해설이 나와야 하므로 picked 가 키에 들어간다
        key = (q["id"], self.picked, self.text.ko)
        if key not in self.banner_cache:
            ch_ = q["choices"][self.picked]
            why = ch_[3] if self.text.ko else ch_[4]
            last = self.is_last_stage(q)
            if self.text.ko:
                head = ("미션 완료!  " if last else "정답!  ") if ok else "오답  "
            else:
                head = ("MISSION CLEAR!  " if last else "Correct!  ") if ok else "Wrong.  "
            # 폭 = 영상 폭 - 양쪽 여백 28, 글자 크기 17, 최대 5줄로 해설 렌더링 (긴 해설은 잘림)
            cov, h = self.text.block(head + why, VID_W - 28, 17, max_lines=5)
            # 배너 높이 = 글 높이 + 위아래 여백 18 (정답이면 아래에 'SPACE / click NEXT' 안내 줄 공간 16 추가)
            self.banner_cache[key] = (cov, h + 18 + (16 if ok else 0))
        cov, bh = self.banner_cache[key]
        # 배너가 시작하는 y. 영상 맨 아래에 붙인다
        y0 = VID_H - bh
        # 배경 색조(BGR): 정답이면 어두운 초록, 오답이면 어두운 빨강
        tint = np.array((30, 95, 30) if ok else (30, 30, 120), np.float32)
        # ROI = 영상의 아래쪽 영역. 슬라이스는 view(참조)라서 아래 roi[:] = ... 가 view 를 직접 수정한다
        roi = view[y0:VID_H]
        # 원본 30% + 색조 70% 로 섞어 반투명 효과. 그 위에 흰 글씨를 얹어도 읽기 쉬워진다
        roi[:] = (roi.astype(np.float32) * 0.30 + tint * 0.70).astype(np.uint8)
        # 해설 글씨(알파 마스크)를 흰색으로 붙인다. 좌표 (14, y0 + 9) = 배너 안쪽 여백
        paste_cov(view, cov, 14, y0 + 9, (255, 255, 255))
        # 보기가 에러를 냈다면 배너 바로 위에 'ERROR: ...' 를 붉게 표시
        if err:
            outlined(view, clip_text("ERROR: " + err, VID_W - 20, 0.45), (10, y0 - 10), 0.45, C_BAD, 1)
        if ok and self.state == "solved":
            put(view, "SPACE / click NEXT", (VID_W - 170, VID_H - 6), 0.45, (255, 255, 255), 1)

    # 문제 아래에 "이미 실행된 코드" 줄(setup)을 작은 글씨로 그린다. 반환값 = 마지막으로 그린 줄 다음의 y.
    #   top   : 시작 y        limit : 이 y 를 넘으면 안 됨 (바로 아래 보기 버튼이 시작하는 위치)
    #   ASCII 줄(코드)   -> cv2.putText (빠름). 패널 폭(432px)에 맞춰 최대 2줄(fit_lines), 줄 간격 13px.
    #                       둘째 줄은 '#' 주석이면 '#   ' 로 시작해서 이어진 줄임을 알 수 있다
    #   한글이 섞인 줄   -> TextRenderer (한글 모드일 때만). 최대 2줄
    def _setup_block(self, p, lines, top, limit):
        for line in lines:
            s = _setup_text(line, self.text.ko)
            # 남은 공간이 한 줄(12px)도 안 되면 그리지 않고 멈춘다 (보기 버튼과 겹침 방지)
            if top > limit - 12:
                break
            # str.isascii(): 모든 문자가 ASCII 면 True (Python 3.7+). cv2.putText 로 그려도 안전한 줄
            if s.isascii():
                subs, sc = fit_lines(s, PANEL_W - 28, SETUP_SCALES, "#   " if s.startswith("#") else "  ")
                for sub in subs:
                    put(p, sub, (14, top + 11), sc, C_ACCENT)
                    top += 13
            # 한글이 들어 있는 줄은 한글 폰트가 있을 때만 그린다 (영문 UI 에서는 영어 튜플 쪽이 이미 ASCII 라서 위 분기로
            #   처리됨)
            elif self.text.ko:
                top = self.text.draw(p, s, 14, top, PANEL_W - 28, 12, C_ACCENT, max_lines=2)
        return top

    # 패널에서 "문제가 바뀔 때만 변하는 정적 부분"을 그려서 캐시한다 (매 프레임 다시 그리지 않음).
    #   - 미션이면 맨 위에 단계 진행 칩(chip) 줄: 완료=초록 / 현재=파랑+테두리 강조 / 남음=회색
    #   - 문제 문장 (최대 4줄) + setup 코드 줄
    #   - 기초 문제는 문장이 시작하는 y 와 setup 시작 y 가 미션과 다르다 (GEO_* 의 첫 버튼 y 위에 맞춰 배치)
    # 캐시 키 = (문제 id, 언어). 언어를 바꾸면 다시 그려야 하므로 키에 포함.
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
            # 단계 칩 하나의 폭: (패널 폭 - 좌우 여백 24 - 칩 사이 간격 합) / 단계 수
            cw = (PANEL_W - 24 - gap * (n - 1)) // n
            for k, stg in enumerate(m["stages"]):
                x0 = 12 + k * (cw + gap)
                y0, y1 = 36, 58
                # done = 이미 끝낸 단계, cur = 지금 단계
                done, cur = k < q["stage_i"], k == q["stage_i"]
                # 칩 배경색(BGR): 완료=초록, 현재=어두운 파랑(95,72,22), 남음=어두운 회색
                fill = C_BTN_RIGHT if done else ((95, 72, 22) if cur else (46, 44, 42))
                cv2.rectangle(p, (x0, y0), (x0 + cw, y1), fill, -1)
                cv2.rectangle(p, (x0, y0), (x0 + cw, y1), C_ACCENT if cur else (90, 90, 90), 1)
                label = "%d %s" % (k + 1, stg["short_ko"] if self.text.ko else stg["short_en"])
                self.text.draw(p, label, x0 + 5, y0 + 2, cw - 8, 13, C_TEXT, max_lines=1)
            # 미션: 칩 줄 아래(y=64)에서 문제 문장 시작. yq = 문장이 끝난 y
            yq = self.text.draw(p, qtext, 14, 64, PANEL_W - 28, 16, C_TEXT, max_lines=4)
            # 문장 바로 아래부터 setup 줄을 그린다. 한계 = 미션 첫 버튼 y(204)
            self._setup_block(p, q["setup"], yq + 4, GEO_MISSION[0])
        else:
            # 기초 문제: 헤더(30px) 아래 y=40 에서 문제 문장 시작 (글자 17px)
            yq = self.text.draw(p, qtext, 14, 40, PANEL_W - 28, 17, C_TEXT, max_lines=4)
            # 기초 문제의 setup 은 최소 y=121 부터 (문장이 짧아도 일정한 위치에 둠). 한계 = 기초 첫 버튼 y(172)
            self._setup_block(p, q["setup"], max(121, yq + 2), GEO_BASIC[0])
        # 완성한 정적 패널을 캐시에 저장 (다음 프레임부터는 복사해서 재사용)
        self.base_cache[key] = p
        return p

    # 오른쪽 패널 한 프레임 = 캐시된 정적 패널 복사본 + 동적 요소(헤더, 보기 버튼, 점수, 다음 버튼).
    # 버튼 색: 정답으로 고른 버튼=초록 / 이미 틀린 버튼=빨강 / 마우스가 올라간 버튼=밝은 회색 / 기본=어두운 회색
    def _panel(self, q):
        # .copy() 필수: 캐시된 원본 위에 직접 그리면 다음 프레임에 이전 프레임의 버튼 색이 남는다
        p = self._panel_base(q).copy()
        # 맨 위 30px 제목 줄 배경 (-1 = 속을 채움)
        cv2.rectangle(p, (0, 0), (PANEL_W, 30), C_HEADER, -1)
        put(p, "Q %d/%d" % (self.qi + 1, len(self.questions)), (12, 21), 0.6, C_WARN, 2)
        put(p, "%s | %s" % (q["unit"], q["unit_name"][:26]), (110, 21), 0.5, C_TEXT)
        n = len(q["choices"])
        # 미션의 마지막 단계를 맞혔다면 보기 버튼 대신 '완성 코드' 박스를 보여 준다
        if self.state == "solved" and self.is_last_stage(q):
            self._clear_box(p, q["mission"])
        else:
            # 보기 5개를 차례로 그린다. 튜플을 풀어 쓰지만 여기서는 code(코드 문자열)만 화면에 쓴다
            for i, (code, fn, ok, wk, we, kind) in enumerate(q["choices"]):
                x0, y0, x1, y1 = btn_rect(i, self.geo)
                # 버튼 색 결정 우선순위: 정답으로 고른 것 > 틀린 것 > 마우스 hover > 기본
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
                # 왼쪽의 어두운 원 + 그 안에 보기 번호(1~5)
                cv2.circle(p, (x0 + 22, (y0 + y1) // 2), 14, (20, 20, 20), -1, cv2.LINE_AA)
                put(p, str(i + 1), (x0 + 16, (y0 + y1) // 2 + 6), 0.6, C_WARN, 2)
                # 코드 문자열은 버튼 안 폭(CODE_W px)에 맞춰 1~2줄로 나눈다 (fit_lines). 2줄이면 위로 9px 올려서 세로 가운데 정렬
                lines, sc = fit_lines(code, CODE_W, CODE_SCALES, "  ")
                ty = (y0 + y1) // 2 - (len(lines) - 1) * 9 + 5
                for ln in lines:
                    put(p, ln, (x0 + CODE_X, ty), sc, C_TEXT)
                    ty += 18
        # 패널 맨 아래: 점수/연속 정답, 정답 후에는 NEXT 버튼, 풀이 중에는 조작 안내
        # 하단 바
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

    # 미션 클리어 시 보기 버튼 영역을 덮는 초록 박스. 내가 정답으로 이어 붙인 '완성 코드'(m['final_lines'])를 보여 준다.
    #   코드가 길어도 한 줄도 잘리지 않도록, 큰 글씨부터 시도해서 '전부 들어가는 첫 크기'를 쓴다 (CLEAR_SCALES).
    #   각 코드 줄은 fit_lines 로 박스 폭(BTN_W - 20 px)에 맞춰 1~2줄로 나눈다.
    def _clear_box(self, p, m):
        y0 = GEO_MISSION[0] - 4
        cv2.rectangle(p, (BTN_X, y0), (BTN_X + BTN_W, 434), (40, 58, 40), -1)
        cv2.rectangle(p, (BTN_X, y0), (BTN_X + BTN_W, 434), C_GOOD, 1)
        put(p, "MISSION CLEAR - your pipeline", (BTN_X + 10, y0 + 20), 0.5, C_GOOD, 1)
        first, last = y0 + 40, 428            # 첫 줄 / 마지막 줄의 baseline y 범위
        for sc in CLEAR_SCALES:
            pitch = int(sc * 37)               # 줄 간격 = 글자 크기에 비례 (0.40 -> 14px, 0.34 -> 12px)
            rows = [fit_lines(ln, BTN_W - 20, (sc,), "  ")[0] for ln in m["final_lines"]]
            n_rows = sum(len(r) for r in rows)
            # 필요한 높이 = (줄 수 - 1) x 줄 간격 + 코드 줄 사이 틈 3px
            if (n_rows - 1) * pitch + 3 * (len(rows) - 1) <= last - first:
                break
        # 공간이 남으면 줄 간격을 최대 18px 까지 넓혀서 읽기 편하게 한다
        pitch = max(pitch, min(18, (last - first - 3 * (len(rows) - 1)) // max(n_rows - 1, 1)))
        y = first
        for r in rows:
            for sub in r:
                put(p, sub, (BTN_X + 10, y), sc, C_TEXT)
                y += pitch
            y += 3

    # 메뉴 화면
    # ---- 메뉴 ----
    # 메뉴 항목 8개의 내용: (제목, 부제목, 사용 가능 여부, 사용 불가 이유) 튜플의 리스트. 순서가 MENU_KINDS 와 같다.
    #   1 기초 / 2~6 미션 1~5 (부제목 = 단계 이름을 ' > ' 로 연결) / 7 전체 도전 (총 문제 수 표시) / 8 자유 실습
    # 미션 5는 손 모델이 없으면 사용 불가로 표시되고 이유(예: 'mediapipe not installed')가 부제목 자리에 나온다.
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

    # 메뉴 패널의 정적 부분을 그려 캐시. 캐시 키 = (언어, 손 사용 가능 여부) -> 둘 중 하나가 바뀌면 다시 그린다
    def _menu_base(self):
        # 아직 손 가능 여부를 모르는 상태(None)는 '가능'으로 간주해서 처음부터 막아 두지 않는다
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

    # 정적 메뉴 + 동적 요소: 마우스가 올라간 항목 테두리(accent 색), 안내 문구(menu_msg; 최대 2줄)
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

    # 결과 요약 화면
    # ---- 결과 ----
    # 세트를 마친 뒤의 결과 화면. 한 번 그리면 캐시(sum_cache)해서 재사용 (세션이 바뀔 때만 None 으로 초기화).
    #   - 기초 문제: 한 줄에 하나씩 '첫 시도 +10 / 2번째 시도 +5 / 정답 +0 / 건너뜀' 라벨
    #   - 미션: 미션당 한 줄, 단계마다 색 점(초록=첫 시도, 노랑=재시도, 빨강=건너뜀) + 획득 점수/만점
    #   - 전부 10점이면 PERFECT!
    #   - 아래에 [R]처음부터 [W]틀린 문제만 [M]메뉴 버튼 (틀린 문제가 없으면 W 는 비활성)
    def _summary_panel(self):
        if self.sum_cache is not None:
            return self.sum_cache
        p = np.full((WIN_H, PANEL_W, 3), C_PANEL, np.uint8)
        cv2.rectangle(p, (0, 0), (PANEL_W, 30), C_HEADER, -1)
        put(p, "RESULT", (12, 21), 0.6, C_WARN, 2)
        # 만점 = 문제 수 x 10
        total = 10 * len(self.questions)
        put(p, "SCORE %d / %d" % (self.score, total), (12, 62), 0.9, C_GOOD, 2)
        y = 92
        # 이미 한 줄로 요약한 미션 id (미션은 단계가 여러 개라 한 줄로 묶기 위해)
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
                    # 단계별 결과 점. 18px 간격으로 가로 배치
                    cv2.circle(p, (262 + k * 18, y - 5), 6, col, -1, cv2.LINE_AA)
                put(p, "+%d/%d" % (pts, 10 * len(stages)), (362, y), 0.5, C_GOOD if pts == 10 * len(stages) else C_WARN, 1)
                y += 22
        # 첫 시도에 못 맞힌(10점 미만) 문제 수 -> W 버튼의 괄호 숫자와 PERFECT 판단에 쓰임
        n_bad = sum(1 for q in self.questions if self.results.get(q["id"], {}).get("pts", 0) < 10)
        if n_bad == 0 and self.questions:
            put(p, "PERFECT!", (14, 396), 0.7, C_GOOD, 2)
        # 버튼 3개 그리기 (rect, 글자, 활성 여부). 비활성 버튼은 어둡게
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


# ======================================================================
# 6. 손 인식 (MediaPipe, 선택 사항)
# ======================================================================
# 세 랜드마크 p1, p2, p3 로 만드는 각 p1-p2-p3 의 크기(도).  p2 가 꼭짓점이다.
#   v1 = p1 - p2,  v2 = p3 - p2   (꼭짓점에서 두 점으로 향하는 벡터)
#   cos(각) = (v1 · v2) / (|v1| |v2|)      <- 내적 공식
#   각 = arccos(cos)  -> 라디안을 도(degrees)로 변환
# x, y 뿐 아니라 z(깊이)도 포함한 3차원 각도다. 손가락이 쭉 펴지면 세 점이 일직선 -> 약 180도, 접을수록 작아진다.
def calculate_angle(p1, p2, p3):
    """노트북 M단원과 같은 방식: p2 를 꼭짓점으로 하는 각도(도)."""
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    # 벡터 길이(노름). 두 점이 겹쳐서 길이가 0이면 각도를 정할 수 없으므로 아래에서 0.0 을 돌려준다
    m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if m1 == 0 or m2 == 0:
        return 0.0
    # 부동소수점 오차로 1.0000001 같은 값이 나오면 arccos 가 NaN 을 내므로 [-1, 1] 로 잘라 준다
    cos = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


# 펴진 손가락 개수: 다섯 손가락(FINGER_ANGLE_POINTS)의 관절 각도가
#   ANGLE_THRESHOLD(160도) 이상인 것을 센다. 자유 실습 모드의 손 인식에 사용
def count_extended_fingers(landmarks):
    n = 0
    for a, b, c in FINGER_ANGLE_POINTS:
        if calculate_angle(landmarks[a], landmarks[b], landmarks[c]) >= ANGLE_THRESHOLD:
            n += 1
    return n


# MediaPipe HandLandmarker 를 감싼 클래스. 핵심 설계: "필요할 때 늦게 불러오기(lazy loading)".
#   - mediapipe 는 import 도 모델 로딩도 느려서, 프로그램 시작 시에는 불러오지 않는다.
#   - available() : 패키지와 모델 파일이 있는지만 가볍게 확인 (메뉴에서 미션5 사용 가능 표시용)
#   - ensure()    : 실제로 한 번만 모델을 로드 (미션5나 손 모드를 처음 열 때)
#   - detect()    : 프레임에서 손 랜드마크를 검출
# mediapipe 가 없거나 모델이 없어도 프로그램은 죽지 않고 해당 기능만 '사용 불가'가 된다.
class HandTracker:
    # model_path: --hand-model 로 직접 지정한 모델 경로 (없으면 기본 위치들을 탐색)
    def __init__(self, model_path=None):
        self.model_path = model_path
        # MediaPipe HandLandmarker 객체 (ensure 후에 생성)
        self.det = None
        # 초기화 성공 여부
        self.ok = False
        # 실패 이유 문자열 (화면에 표시됨)
        self.err = ""
        # ensure() 를 이미 시도했는가 (실패해도 같은 시도를 반복하지 않기 위한 플래그)
        self.tried = False
        # 타임스탬프 기준 시각
        self.t0 = time.monotonic()
        # 마지막으로 사용한 타임스탬프(ms). VIDEO 모드는 타임스탬프가 반드시 증가해야 해서 기록해 둔다
        self.last_ts = -1

    # 모델 파일(.task) 위치 탐색 순서: --hand-model 경로 -> 기본 상대 경로 -> 이 파일 기준 경로 -> 한 단계 위 -> 이 파일 옆의
    #   hand_landmarker.task. 처음 존재하는 파일을 돌려준다
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

    # MediaPipe 패키지와 모델 파일이 있는지만 가볍게 확인한다 (모델을 로드하지 않고 find_spec 으로 설치 여부만 검사 -> 빠름).
    #   반환: (가능 여부, 이유)
    def available(self):
        """MediaPipe 와 모델 파일이 있는지 가볍게 확인 (모델은 아직 불러오지 않는다)."""
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

    # 모델을 실제로 로드한다. (한 번만 시도. 결과는 self.ok / self.err 에 저장)
    # 옵션 설명:
    #   running_mode = VIDEO : 연속된 프레임을 처리하는 모드. 이전
    #     프레임의 결과를 이용해 추적하므로 IMAGE 모드보다 빠르고 안정적
    #                          (대신 detect_for_video 에 '증가하는 타임스탬프'를 줘야 함)
    #   num_hands = 2        : 최대 두 손까지 검출
    #   min_*_confidence = 0.5 : 검출 / 존재 / 추적 신뢰도 기준 (낮추면 더 잘 잡히지만 오검출 증가)
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

    # BGR 프레임 -> 손 랜드마크 목록. 반환: 손마다 21개 랜드마크 리스트 (없으면 빈 리스트).
    # 각 랜드마크는 p.x, p.y (0~1 정규화 좌표), p.z (깊이; 손목 기준 상대값) 를 가진다.
    #   1) BGR -> RGB 변환 : MediaPipe 의 ImageFormat.SRGB 는 R, G, B 순서를 기대한다 (미션5 1단계의 정답이 이것)
    #   2) np.ascontiguousarray : 메모리가 연속인 배열이어야 MediaPipe 가 받을 수 있다
    #   3) 타임스탬프(ms) : 이전보다 최소 1 커야 함 -> max(last_ts + 1, 경과 ms)
    def detect(self, frame_bgr):
        if not self.ensure():
            return []
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        ts = max(self.last_ts + 1, int((time.monotonic() - self.t0) * 1000))
        self.last_ts = ts
        return self.det.detect_for_video(image, ts).hand_landmarks

    # 프로그램 종료 시 MediaPipe 자원을 해제
    def close(self):
        if self.det is not None:
            self.det.close()


# ======================================================================
# 7. 자유 실습 모드
# ======================================================================
# 자유 실습 모드의 설정 표들 (키로 값을 순환하며 바꾼다).
#
# SEG_PRESETS : 9번 키로 순환하는 색 분할 프리셋. (이름, [(HSV 하한, HSV 상한), ...])
#    - 한 프리셋에 범위가 여러 개면 각각 inRange 한 뒤 OR 로 합친다.
#    - 'red (2 ranges)' 가 범위 2개인 이유: HSV 에서 빨강은 색상(H)이 0 근처와 179 근처로 "양 끝"에 걸쳐 있다
#      (색상환이 원형이라 0 과 179 가 이어져 있음). 그래서 H 0~8 과 172~179 를 따로 잡아 합쳐야 한다.
#    - ('off', None) 은 분할을 끈 상태.
SEG_PRESETS = [
    ("off", None),
    ("skin", [((0, 40, 70), (20, 255, 255))]),
    ("green", [((40, 60, 50), (85, 255, 255))]),
    ("orange", [((8, 100, 100), (25, 255, 255))]),
    ("red (2 ranges)", [((0, 100, 70), (8, 255, 255)), ((172, 100, 70), (179, 255, 255))]),
    ("blue", [((95, 80, 50), (130, 255, 255))]),
]
# 1번 키: 반전 모드 이름과 실제 flipCode (None = 끔, 1 = 좌우, 0 = 상하, -1 = 둘 다). 인덱스가 서로 대응한다
FLIP_NAMES = ["off", "LR  flip(.,1)", "UD  flip(.,0)", "BOTH flip(.,-1)"]
FLIP_CODES = [None, 1, 0, -1]
# 6번 키: 샤프닝 커널 종류 (끔 / 5 커널 / 9 커널 / 약한 커널). SHARP_KS 의 같은 인덱스가 실제 커널 행렬
SHARP_NAMES = ["off", "kernel 5 (4-neighbor)", "kernel 9 (8-neighbor)", "kernel 3 (weak)"]
SHARP_KS = [None, SHARP_K, SHARP_K8, SHARP_KW]
# 7번 키: 엣지 검출 방식 (끔 / Sobel(기울기 크기) / Canny)
EDGE_NAMES = ["off", "Sobel", "Canny"]
# 3번 키: 보여 줄 HSV 채널 (끔 / H 색상 / S 채도 / V 명도). 채널 하나만 흑백으로 보여 준다
HSV_NAMES = ["off", "H", "S", "V"]
# [ ] 키로 조절할 수 있는 파라미터. 각 항목: v = 현재 값, lo / hi = 허용 범위, step = 한 번에 바꾸는 양.
#   box, gauss : 블러 커널 크기 (홀수여야 하므로 시작값 11, 최소 1, step 2 -> 항상 홀수 유지)
#   clip       : CLAHE 의 clipLimit (대비 증폭 한도)
#   canny      : Canny 낮은 임계값 (높은 임계값은 항상 낮은 값의 2배로 자동 설정)
#   ahead      : Kalman 예측 화살표가 몇 프레임 앞을 가리킬지
PARAMS = {
    "box": dict(v=11, lo=1, hi=51, step=2),
    "gauss": dict(v=11, lo=1, hi=51, step=2),
    "clip": dict(v=3.0, lo=1.0, hi=10.0, step=0.5),
    "canny": dict(v=100, lo=10, hi=250, step=10),
    "ahead": dict(v=8, lo=1, hi=30, step=1),
}


# 자유 실습 모드: 퀴즈 없이 키를 눌러 효과를 켜고 끄며 실시간으로 실험한다.
#  - 켜 둔 효과는 process() 에서 정해진 순서로 차례로 적용되고, 지금 실행 중인 코드가 패널에 한 줄씩 표시된다
#    (self.codes) -> 키를 누르는 것과 코드의 대응 관계를 배울 수 있다.
#  - 퀴즈 모드와 HandTracker 객체를 공유한다 (App 에서 전달).
class FreePlay:
    # hand_model: --hand-model 로 지정한 손 모델 경로 (없으면 기본 위치 탐색)
    def __init__(self, hand_model=None):
        self.hands = HandTracker(hand_model)
        self.reset()

    # 모든 효과를 끄고 파라미터를 기본값으로 되돌린다 (R 키). 시작할 때도 호출
    def reset(self):
        # 효과 상태 변수들:
        #   flip   : FLIP_NAMES 인덱스            gray : 흑백 on/off          hsvch : HSV 채널 인덱스
        #   box / gauss : 블러 on/off             sharp : 샤프닝 종류 인덱스    clahe : CLAHE on/off
        #   edge   : 엣지 방식 인덱스             seg  : 색 분할 프리셋 인덱스   morph : 모폴로지 on/off
        #   contour / harris / kalman / hand / pip : 각 기능 on/off
        #   kt     : Kalman 추적 대상 'color'(색 덩어리 중심) 또는 'finger'(검지 끝)
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
        # 파라미터 표를 깊은 복사: dict(v) 로 각 항목을 새로 만들어야 reset 후에도 PARAMS 원본 값이 오염되지 않는다
        self.p = {k: dict(v) for k, v in PARAMS.items()}
        # [ ] 키가 현재 조절하는 파라미터 이름 (예: 'gauss'). 효과를 켜면 해당 파라미터가 자동 선택된다
        self.sel = None
        # Kalman 필터 객체와 연속으로 대상을 놓친 프레임 수
        self.kf = None
        self.kf_lost = 0
        # 추정 위치 궤적 (최근 40개)
        self.trail = deque(maxlen=40)
        # 화면에 잠깐 띄우는 안내 문구와 사라질 시각 (say 참고)
        self.msg = ""
        self.msg_until = 0.0
        # '지금 실행 중인 코드' 줄들. process() 가 켜진 효과마다 한 줄씩 추가하고 패널이 마지막 6줄을 보여 준다
        self.codes = []
        # 자유 모드 전용 CLAHE 객체 (clipLimit 을 [ ] 키로 바꿀 수 있어서 전역 _CLAHE 와 분리)
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # 화면에 3초 동안 안내 문구를 띄운다 (예: 'MediaPipe unavailable')
    def say(self, text):
        self.msg = text
        self.msg_until = time.monotonic() + 3.0

    # 키 입력 -> 상태 변경. 숫자키 1~9 와 M, C, H, K, T, G, V, R, [ ] 를 처리한다
    # ---- 키 ----
    # k: cv2.waitKey 키 코드. 효과를 토글(on/off)하거나 순환(0->1->2->...)시키고, 조절 대상(self.sel)을 지정
    def key(self, k):
        c = chr(k).lower() if 0 <= k < 128 else ""
        if c == "1":
            self.flip = (self.flip + 1) % 4
        elif c == "2":
            self.gray = not self.gray
        # 순환형: 누를 때마다 (현재 + 1) % 개수 -> 끔 -> H -> S -> V -> 끔 ...
        elif c == "3":
            self.hsvch = (self.hsvch + 1) % 4
        # 토글형: not 로 on/off 를 뒤집는다. 켜면 [ ] 로 조절할 파라미터를 box 로 지정
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
        # Kalman 토글: 켤 때마다 필터를 새로 만들고(kf=None) 궤적을 비운다
        elif c == "k":
            self.kalman = not self.kalman
            self.kf = None
            self.trail.clear()
            self.sel = "ahead" if self.kalman else self.sel
        # Kalman 추적 대상 전환: 색 덩어리 <-> 검지 끝. 검지 끝은 MediaPipe 가
        #   필요해서 ensure() 가 실패하면 안내 문구만 띄우고 전환하지 않는다
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
        # 손 인식 토글: 켤 때 모델 로드에 실패하면 안내 문구를 띄우고 다시 끈다
        elif c == "g":
            self.hand = not self.hand
            if self.hand and not self.hands.ensure():
                self.say("MediaPipe unavailable: " + self.hands.err)
                self.hand = False
        elif c == "v":
            self.pip = not self.pip
        elif c == "r":
            self.reset()
        # 파라미터 조절: ']' 는 + step, '[' 는 - step. min/max 로 [lo, hi] 범위 안으로 제한
        elif c in ("[", "]") and self.sel:
            sp = self.p[self.sel]
            sp["v"] = min(sp["hi"], max(sp["lo"], sp["v"] + (sp["step"] if c == "]" else -sp["step"])))

    # 영상 처리 메서드들
    # ---- 처리 ----
    # 색 분할 마스크를 만든다.  seg 가 꺼져 있어도(0) 윤곽/Kalman 용 마스크가 필요하면 'skin' 프리셋(1번)을 기본으로 쓴다.
    # HSV 변환 후 프리셋의 각 범위로 inRange -> OR 로 합침 -> (morph 가 켜져 있으면) OPEN 으로 점 제거 + CLOSE 로 구멍 메움.
    def seg_mask(self, img_bgr):
        # 프리셋 선택. self.seg == 0(끔)일 때는 인덱스 1 = 'skin'
        name, ranges = SEG_PRESETS[self.seg] if self.seg else SEG_PRESETS[1]
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = None
        for lo, hi in ranges:
            m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        # 모폴로지 정리: OPEN(침식->팽창) 은 흰 점 제거, CLOSE(팽창->침식) 은 흰 덩어리 속 검은 구멍 메움. 순서가
        #   중요하다(점을 먼저 지우고 구멍을 메움)
        if self.morph:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    # 인스턴스 상태를 쓰지 않는 보조 함수라서 staticmethod. 미션의 _skin_blob/_contours 와 같은 규칙(가장 큰 윤곽, 최소 면적)
    @staticmethod
    def largest_contour(mask, min_area=400):
        res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # OpenCV 3.x(3개 반환) / 4.x·5.x(2개 반환) 모두 지원
        cnts = res[0] if len(res) == 2 else res[1]
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        return c if cv2.contourArea(c) >= min_area else None

    # 한 프레임의 Kalman 처리. meas = (x, y) 측정값 또는 None. 반환: (x, y, vx, vy) 추정값 또는 None(추적 중 아님).
    # 미션4 의 _make_kf / kalman_frame 과 같은 등속도 모델이다 (상태 [x, y, vx, vy], 측정 [x, y]).
    #   - 필터가 없고 측정도 없으면 아무것도 안 함
    #   - 첫 측정에서 필터를 만들고 그 위치에서 시작
    #   - 측정이 있으면 predict -> correct, 없으면 predict 결과(pred)를 그대로 추정으로 사용
    #   - 25프레임 넘게 못 보면 추적 종료(필터 삭제)
    def _kalman_step(self, meas):
        if self.kf is None:
            if meas is None:
                return None
            # 행렬 의미는 _make_kf 의 주석 참고 (F=등속도 전이, H=위치만 측정, Q/R/P=잡음과 초기 불확실성)
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
            if self.kf_lost > 25:        # 너무 오래 안 보이면 추적 종료
                self.kf = None
                self.trail.clear()
                return None
        return float(est[0, 0]), float(est[1, 0]), float(est[2, 0]), float(est[3, 0])

    # 한 프레임에 켜 둔 효과를 정해진 순서로 적용해서 결과 영상을 돌려준다.
    #
    # 적용 순서 (앞의 결과가 뒤의 입력):
    #   반전 -> 흑백 -> HSV 채널 -> box 블러 -> Gaussian 블러 -> 샤프닝 -> CLAHE -> 엣지
    #   -> (색 분할 마스크 적용) -> 윤곽/중심 -> Harris 코너 -> 손 인식 -> Kalman -> 원본 PIP
    #
    # 주의할 점:
    #   * 마스크 / 윤곽 / Harris / 손 인식은 "반전만 된 원본(base)"으로
    #     계산한다. 블러나 엣지 효과가 검출 결과를 망가뜨리지 않게 하려는 것
    #     (그림은 효과가 적용된 img 위에 그린다).
    #   * 효과를 켤 때마다 self.codes 에 대응하는 코드 한 줄을 넣는다 -> 패널의 'CODE RUNNING' 에 표시됨.
    def process(self, raw):
        self.codes = []
        # base: 반전만 적용한 원본 (검출용 기준 영상). 반전이 꺼져 있으면 raw 그대로
        base = raw if self.flip == 0 else cv2.flip(raw, FLIP_CODES[self.flip])
        if self.flip:
            self.codes.append("frame = cv2.flip(frame, %d)" % FLIP_CODES[self.flip])
        # img: 효과를 누적해서 적용할 작업 영상 (base 를 보호하려고 복사)
        img = base.copy()
        h, w = img.shape[:2]
        # 흑백으로 만들었다가 다시 BGR(3채널)로 되돌린다. 이후 단계(블러, 그리기 등)가 3채널을 가정하기 때문. 화면에는
        #   흑백으로 보인다
        if self.gray:
            img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            self.codes.append("gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)")
        # HSV 로 바꾼 뒤 선택한 채널(H/S/V) 하나만 꺼내 흑백으로 보여 준다
        if self.hsvch:
            ch = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))[self.hsvch - 1]
            if self.hsvch == 1:
                # H 채널은 최댓값이 179 라서 그대로 보면 어둡다 -> 255/179 를 곱해 0~255 로 늘려 밝게 표시 (표시용 변환일 뿐
                #   값의 의미는 그대로)
                ch = (ch.astype(np.float32) * (255.0 / 179.0)).astype(np.uint8)  # H(0~179) -> 0~255 로 늘려 표시
            img = cv2.cvtColor(ch, cv2.COLOR_GRAY2BGR)
            self.codes.append("h, s, v = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))   # show %s" % HSV_NAMES[self.hsvch].lower())
        # box 블러: cv2.blur 의 커널 크기 k x k (k 는 [ ] 키로 조절)
        if self.box:
            k = int(self.p["box"]["v"])
            img = cv2.blur(img, (k, k))
            self.codes.append("frame = cv2.blur(frame, (%d, %d))" % (k, k))
        # Gaussian 블러: sigma=0 이면 커널 크기에서 자동 계산
        if self.gauss:
            k = int(self.p["gauss"]["v"])
            img = cv2.GaussianBlur(img, (k, k), 0)
            self.codes.append("frame = cv2.GaussianBlur(frame, (%d, %d), 0)" % (k, k))
        # 샤프닝: 종류별 커널로 filter2D. -1 은 입력과 같은 자료형(uint8) 출력
        if self.sharp:
            img = cv2.filter2D(img, -1, SHARP_KS[self.sharp])
            self.codes.append("frame = cv2.filter2D(frame, -1, sharp_kernel)   # %s" % SHARP_NAMES[self.sharp])
        # CLAHE: LAB 의 L 채널에만 적용. clipLimit 은 [ ] 로 바꾼 값을 setClipLimit 으로 반영
        if self.clahe:
            clip = float(self.p["clip"]["v"])
            self._clahe.setClipLimit(clip)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            img = cv2.cvtColor(cv2.merge([self._clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
            self.codes.append("clahe = cv2.createCLAHE(clipLimit=%.1f, tileGridSize=(8, 8))   # on L" % clip)
        # 엣지: 먼저 흑백으로 변환. Sobel 은 x, y 방향 미분의 크기(magnitude)를
        #   /4 로 줄여 0~255 에 맞춤, Canny 는 (낮은 임계값, 2배)
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

        # 색 분할 / 윤곽 / Kalman(색 대상)에 마스크가 필요할 때만 계산한다 (불필요한 연산 방지)
        mask = None
        need_mask = self.seg or self.contour or (self.kalman and self.kt == "color")
        if need_mask:
            mask = self.seg_mask(base)
        if self.seg:
            img = cv2.bitwise_and(img, img, mask=mask)
            self.codes.append("mask = cv2.inRange(hsv, lower, upper)   # %s%s" % (
                SEG_PRESETS[self.seg][0], " + open/close" if self.morph else ""))
        # blob: 윤곽 중심(무게중심) 좌표. 윤곽 표시나 Kalman(색 대상)의 측정값으로 사용
        blob = None
        if self.contour or (self.kalman and self.kt == "color"):
            cnt = self.largest_contour(mask)
            if cnt is not None:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    blob = (M["m10"] / M["m00"], M["m01"] / M["m00"])
                if self.contour:
                    cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
                    # 윤곽을 감싸는 최소 원(주황). 중심 (cx, cy) 와 반지름 r
                    (cx, cy), r = cv2.minEnclosingCircle(cnt)
                    cv2.circle(img, (int(cx), int(cy)), int(r), (0, 200, 255), 2)
                    if blob:
                        cv2.circle(img, (int(blob[0]), int(blob[1])), 5, (0, 0, 255), -1)
            if self.contour:
                self.codes.append("cnt = max(contours, key=cv2.contourArea); M = cv2.moments(cnt)")
        # Harris 코너: 흑백 float32 -> cornerHarris -> dilate -> 최댓값의
        #   1% 보다 큰 곳을 빨강으로 표시 (미션3 정답과 같은 방식)
        if self.harris:
            g32 = np.float32(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY))
            dst = cv2.dilate(cv2.cornerHarris(g32, 2, 3, 0.04), None)
            if dst.max() > 0:
                img[dst > 0.01 * dst.max()] = (0, 0, 255)
            self.codes.append("dst = cv2.dilate(cv2.cornerHarris(np.float32(gray), 2, 3, 0.04), None)")

        # 손 랜드마크: 손 인식이 켜져 있거나 Kalman 대상이 '검지 끝'일 때만 검출 (MediaPipe 호출은 비싸다)
        hand_lm = None
        if self.hand or (self.kalman and self.kt == "finger"):
            hand_lm = self.hands.detect(base)
        if self.hand and hand_lm:
            for lm in hand_lm:
                # 정규화 좌표(0~1) -> 픽셀 좌표 (미션5 2단계의 정답 변환)
                pts = [(int(p.x * w), int(p.y * h)) for p in lm]
                for a, b in HAND_CONNECTIONS:
                    cv2.line(img, pts[a], pts[b], (0, 255, 0), 2)
                for i, pt in enumerate(pts):
                    cv2.circle(img, pt, 5 if i in (4, 8, 12, 16, 20) else 3,
                               (0, 0, 255) if i in (4, 8, 12, 16, 20) else (255, 0, 0), -1)
                # 관절 각도 >= 160도 인 손가락 수를 세어 손목 근처에 표시
                n = count_extended_fingers(lm)
                outlined(img, "Fingers: %d" % n, (pts[0][0] - 30, pts[0][1] + 30), 0.7, C_WARN)
            self.codes.append("extended = calculate_angle(lm[a], lm[b], lm[c]) >= %d" % ANGLE_THRESHOLD)
        elif self.hand:
            self.codes.append("# no hand detected")

        # Kalman 추적. 측정값 meas:
        #   kt == 'finger' -> 검지 끝(랜드마크 8번)의 픽셀 좌표       kt == 'color' -> 색 덩어리의 무게중심(blob)
        # 표시: 빨강 원 = 측정, 초록 점 = 필터 추정, 청록 선 = 궤적, 노랑 화살표 = 속도 x ahead 프레임 앞 위치
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
                cv2.circle(img, (int(meas[0]), int(meas[1])), 7, (0, 0, 255), 2)          # 측정값: 빨강
            if est is not None:
                ex, ey, vx, vy = est
                self.trail.append((int(ex), int(ey)))
                if len(self.trail) > 1:
                    cv2.polylines(img, [np.array(self.trail, np.int32)], False, (255, 255, 0), 2)
                cv2.circle(img, (int(ex), int(ey)), 6, (0, 255, 0), -1)                   # 필터 추정: 초록
                k = int(self.p["ahead"]["v"])
                fx, fy = int(ex + vx * k), int(ey + vy * k)
                cv2.arrowedLine(img, (int(ex), int(ey)), (fx, fy), (0, 255, 255), 3, tipLength=0.25)  # 예측: 노랑
                lost = "   (target lost: predicting)" if meas is None else ""
                outlined(img, "Kalman%s" % lost, (10, h - 12), 0.55, C_WARN, 1)
            self.codes.append("est = kf.predict(); kf.correct(z)   # z=%s, arrow=%d frames ahead" % (
                self.kt, int(self.p["ahead"]["v"])))

        # PIP(화면 속 화면): 효과가 적용되지 않은 원본(raw)을 오른쪽 아래에 200x150 으로 작게 보여 준다 -> 전/후 비교
        if self.pip:
            small = cv2.resize(raw, (200, 150), interpolation=cv2.INTER_AREA)
            img[h - 160:h - 10, w - 210:w - 10] = small
            cv2.rectangle(img, (w - 210, h - 160), (w - 10, h - 10), (255, 255, 255), 1)
        return img

    # 자유 모드 화면 그리기
    # ---- 그리기 ----
    # 왼쪽 = process() 결과 영상, 오른쪽 = 키 안내 패널. 안내 문구(msg)는 사라질 시각 전까지만 표시
    def render(self, raw, now):
        img = self.process(raw)
        canvas = np.zeros((WIN_H, WIN_W, 3), np.uint8)
        canvas[:, :VID_W] = img
        canvas[:, VID_W:] = self._panel()
        outlined(canvas, "FREE PLAY", (10, 24), 0.6, C_ACCENT)
        if self.msg and time.monotonic() < self.msg_until:
            outlined(canvas, self.msg[:70], (10, VID_H - 34), 0.45, C_WARN, 1)
        return canvas

    # 오른쪽 패널: 각 효과의 키 / 이름 / 현재 값 목록 (켜져 있으면 초록, 꺼져 있으면 회색), [ ] 조절 대상, 'CODE RUNNING'
    #   코드 줄 (최근 6개). 괄호 속 알파벳은 노트북 단원 표시
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
        # 목록 시작 y. 한 줄 19px 간격 (16줄 -> y = 52 ~ 337)
        #   이전에는 20px 간격이라 마지막 코드 줄이 화면 아래(y=482 > 480)로 잘렸다
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
        # 최근 코드 6줄 (위 계산으로 마지막 줄 baseline 이 약 460). 한 줄이 패널 폭(436px)을 넘으면 잘라서 그린다
        for line in self.codes[-6:]:
            put(p, line[:72], (12, y), 0.33, C_TEXT)
            y += 13
        return p


# ======================================================================
# 8. 앱 / 메인 루프
# ======================================================================
# 화면 해상도 (너비, 높이) 를 구한다. 모르면 None.
#   Windows : ctypes 로 user32.GetSystemMetrics(0, 1) 호출 (너비, 높이) - 추가 설치 불필요
#   Linux   : 1) xrandr / xdpyinfo 명령 결과에서 읽는다 (Jetson Ubuntu 기본 도구, 추가 설치 불필요)
#             2) 안 되면 tkinter 로 임시 창을 만들어 화면 크기를 읽고 바로 닫는다. (python3-tk 가 없거나 화면이 없으면 예외 -> None)
def _screen_size_x11():
    """xrandr / xdpyinfo 출력에서 화면 해상도를 읽는다 (X11). 실패하면 None."""
    import subprocess
    for cmd in (["xrandr", "--current"], ["xdpyinfo"]):
        try:
            out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=3).stdout
            out = out.decode("utf-8", "replace")
        except Exception:
            continue
        # xrandr 줄 예:  "HDMI-0 connected primary 1920x1080+0+0 (normal ...)"  -> 주 모니터 해상도를 우선 사용
        # \bconnected 는 "disconnected" 에는 맞지 않는다 (단어 경계가 없으므로)
        for pat in (r"\bconnected primary (\d+)x(\d+)\+", r"\bconnected (\d+)x(\d+)\+",
                    r"current\s+(\d+)\s*x\s*(\d+)", r"dimensions:\s+(\d+)x(\d+)"):
            m = re.search(pat, out)
            if m:
                return int(m.group(1)), int(m.group(2))
    return None


def screen_size():
    """화면 해상도 (알 수 없으면 None)."""
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


# 창 확대 배율을 자동으로 정한다.  기준 = 화면 너비의 96% 안에 논리 캔버스(1100 px)가 들어가고,
# 높이는 작업표시줄 / 제목 줄 여유 130px 를 뺀 값 안에 480 px 가 들어가도록 하는 배율 중 작은 값.
#  -> 0.05 단위로 반올림, 범위 1.0 ~ 2.2.   예) 1920x1080 -> 1.7배,  1366x768 -> 1.2배.
# 해상도를 모르면 1.0배 (1100x480 은 어떤 모니터에서도 들어간다. 키우려면 + 키).
def auto_scale():
    """화면에 꽉 차게(작업표시줄 여유 포함) 창 배율을 정한다. 해상도를 모르면 1.0배."""
    sz = screen_size()
    if not sz:
        return 1.0
    sw, sh = sz
    # 가로 기준 배율과 세로 기준 배율 중 작은 쪽을 쓴다 (어느 쪽도 화면을 넘지 않도록)
    s = min(sw * 0.96 / WIN_W, (sh - 130) / WIN_H)
    # round(s * 20) / 20 = 0.05 단위 반올림. 최소 1.0(축소 안 함), 최대 2.2 로 제한
    return max(1.0, min(2.2, round(s * 20) / 20.0))


# 프로그램 전체를 묶는 클래스: 퀴즈 모드와 자유 모드를 하나로 관리하고, 키 / 마우스 입력과 창 배율을 처리한다.
# 좌표 규칙: 모든 그리기와 버튼 판정은 "논리 좌표(1100x480)"에서 하고, 마지막에 present() 가 확대해서 보여 준다.
# 마우스 좌표는 mouse_cb 에서 배율로 나눠서 논리 좌표로 되돌린다.
class App:
    # args: parse_args 의 결과. 퀴즈와 자유 모드가 같은 HandTracker(self.free.hands)를 공유한다
    def __init__(self, args):
        self.text = TextRenderer(args.lang)
        self.free = FreePlay(args.hand_model)
        self.quiz = Quiz(self.text, args.seed, hands=self.free.hands, start=getattr(args, "set", "menu"))
        # 현재 모드: 'quiz' 또는 'free' (P 키로 전환)
        self.mode = args.mode
        # --scale 값이 있으면 사용, 0(기본)이면 auto_scale() 로 화면에 맞춘다
        sc = getattr(args, "scale", 0) or 0
        self.scale = float(sc) if sc > 0 else auto_scale()
        # 배율을 0.8 ~ 3.0 범위로 제한
        self.scale = max(0.8, min(3.0, self.scale))
        # FPS 표시용 이동평균 값과 직전 프레임 시각
        self.fps = 0.0
        self._t_prev = None

    # 현재 모드의 화면(논리 캔버스)을 만들고 FPS 를 얹는다. FPS 는 지수 이동평균(0.9 이전 + 0.1 현재)으로 부드럽게 표시
    def render(self, frame, now):
        if self._t_prev is not None:
            dt = max(now - self._t_prev, 1e-3)
            # 처음에는 이전 값이 없으므로 현재 순간값(1/dt)으로 시작
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt
        self._t_prev = now
        canvas = self.quiz.render(frame, now) if self.mode == "quiz" else self.free.render(frame, now)
        outlined(canvas, "%.0f FPS" % self.fps, (10, 46), 0.45, C_TEXT, 1)
        return canvas

    # 논리 캔버스를 창 배율만큼 키워서 돌려준다 (imshow 직전 단계).
    #  - 배율 1.0 이면 그대로 반환 (복사 없음)
    #  - 확대(>1)는 INTER_LINEAR(부드럽게), 축소(<1)는 INTER_AREA(계단 현상이 적음)
    # WINDOW_AUTOSIZE 창은 이미지 크기를 그대로 창 크기로 쓰므로, 확대한 이미지를 보여 주면 창도 그만큼 커진다.
    def present(self, canvas):
        """논리 화면(1100x480)을 창 배율만큼 키워서 보여 준다."""
        s = self.scale
        if abs(s - 1.0) < 0.01:
            return canvas
        interp = cv2.INTER_LINEAR if s > 1.0 else cv2.INTER_AREA
        return cv2.resize(canvas, (int(round(WIN_W * s)), int(round(WIN_H * s))), interpolation=interp)

    # 배율을 0.05 단위로 맞추고 0.8~3.0 으로 제한한 뒤 터미널에 알린다 (+ / - / 0 키에서 호출)
    def set_scale(self, s):
        self.scale = max(0.8, min(3.0, round(s * 20) / 20.0))
        print("[ui] window scale x%.2f" % self.scale)

    # Quiz 가 돌려준 동작을 처리: 'free' 면 자유 실습 모드로 전환 (메뉴 8번)
    def _act(self, action):
        if action == "free":
            self.mode = "free"

    # 공통 키를 먼저 처리하고, 나머지는 현재 모드에 넘긴다.  반환값 False = 프로그램 종료.
    #   Q / ESC(27) 종료,  P 모드 전환,  + / = 확대,  - / _ 축소,  0 자동 크기
    def key(self, k, now):
        """False 를 돌려주면 종료."""
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

    # OpenCV 마우스 콜백. 퀴즈 모드에서만 동작. 이동은 hover 표시, 왼쪽 버튼 누름은 클릭 처리
    def mouse_cb(self, event, x, y, flags, param):
        if self.mode != "quiz":
            return
        # 창(확대된) 좌표 -> 논리 좌표: 배율로 나눈다. 이걸 빼먹으면 확대했을 때 클릭 위치가 어긋난다
        x, y = int(x / self.scale), int(y / self.scale)       # 화면 좌표 -> 논리 좌표
        if event == cv2.EVENT_MOUSEMOVE:
            self.quiz.move(x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self._act(self.quiz.click(x, y, time.monotonic()))

    # 프로그램 종료 시 MediaPipe 자원 해제
    def close(self):
        self.free.hands.close()


# 명령행 옵션 정의 (파일 맨 위 docstring 에 사용 예가 있다). argv=None
#   이면 sys.argv 를 사용, 리스트를 주면 테스트에서 직접 호출 가능
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="AirBall Vision Quiz Lab (06_03, Jetson version) - no options needed")
    ap.add_argument("--csi", action="store_true", help="Jetson CSI camera (nvarguscamerasrc) instead of the USB camera")
    ap.add_argument("--video", help="use a video file instead of a camera")
    ap.add_argument("--image", help="use a still image instead of a camera")
    ap.add_argument("--demo", action="store_true", help="synthetic video, no camera needed")
    ap.add_argument("--mode", choices=["quiz", "free"], default="quiz")
    # --set: 메뉴를 건너뛰고 바로 시작할 세트 (basic / missions / all / m1~m5)
    ap.add_argument("--set", choices=["menu", "basic", "missions", "all", "m1", "m2", "m3", "m4", "m5"],
                    default="menu", help="skip the menu and start this question set")
    ap.add_argument("--scale", type=float, default=0.0,
                    help="window size multiplier (default: fit the screen automatically; +/- keys change it live)")
    ap.add_argument("--lang", choices=["auto", "ko", "en"], default="auto")
    # --seed: 보기 순서 섞기를 재현하고 싶을 때 (같은 값이면 같은 순서)
    ap.add_argument("--seed", type=int, default=None, help="shuffle seed for the choices")
    ap.add_argument("--hand-model", default=None, help="path to hand_landmarker.task")
    return ap.parse_args(argv)


# 모니터에 연결된 터미널인지 확인한다. Linux 에서 DISPLAY(X11) 도 WAYLAND_DISPLAY 도 없으면 cv2.imshow 가 창을 못 만들고
# 이해하기 어려운 GTK 오류로 죽는다 (SSH 로 접속했을 때 흔하다). 그 전에 해결 방법을 알려 준다.
def check_display():
    if platform.system() != "Linux":
        return True
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return True
    print("[error] 화면(DISPLAY)을 찾을 수 없어 창을 띄울 수 없습니다.")
    print("        Jetson 에 모니터가 연결돼 있다면 그 데스크톱의 터미널에서 실행하세요.")
    print("        SSH 로 접속한 상태라면:  export DISPLAY=:0   를 입력한 뒤 다시 실행하면 Jetson 모니터에 창이 뜹니다.")
    return False


# 카메라 프레임을 연달아 이만큼 못 읽으면 종료한다. 실패할 때마다 30ms 쉬므로 약 3초.
# (카메라가 막 켜진 직후에는 첫 프레임이 늦게 오는 경우가 있어서 바로 종료하지 않는다)
READ_FAIL_LIMIT = 100


# 프로그램 진입점. 순서:
#   1) 옵션 해석 + 화면(DISPLAY) 확인  2) 영상 입력(FrameSource)과 App 생성  3) 창 생성 + 마우스 콜백 등록
#   4) 메인 루프: 프레임 읽기 -> 화면 만들기 -> 확대해서 표시 -> 키 확인   (Q/ESC 나 프레임 읽기 실패 시 종료)
#   5) 어떤 경우든 finally 에서 카메라와 창, MediaPipe 를 정리
def main(argv=None):
    args = parse_args(argv)
    if not check_display():
        return 1
    # 카메라 / 파일을 못 열면 긴 traceback 대신 이유(와 해결 방법)만 보여 주고 종료한다
    try:
        src = FrameSource(args)
    except RuntimeError as e:
        print("[error] %s" % e)
        return 1
    app = App(args)
    print("[ui] Korean font: %s" % (app.text.font_path or "not found -> English UI  (Korean: sudo apt install fonts-nanum)"))
    print("[ui] window scale x%.2f  (+ / - / 0 to change)" % app.scale)
    # WINDOW_AUTOSIZE: 창 크기가 이미지 크기에 자동으로 맞춰진다 (사용자가 마우스로 창 크기를 바꿀 수 없음 -> 마우스 좌표와
    #   이미지 픽셀이 1:1 로 대응)
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
    try:
        # 창을 화면 왼쪽 위(10, 10)에 놓는다. 일부 환경에서는 창 위치 지정이 지원되지 않아 예외가 날 수 있으므로 try 로
        #   감쌌다
        cv2.moveWindow(WIN_NAME, 10, 10)
    except Exception:
        pass
    # 이 창에서 일어나는 마우스 이벤트를 app.mouse_cb 로 보내도록 등록
    cv2.setMouseCallback(WIN_NAME, app.mouse_cb)
    rc = 0          # 종료 코드: 0 정상, 1 카메라 프레임을 못 읽어서 종료
    fails = 0       # 연속으로 프레임 읽기에 실패한 횟수
    warned = False  # "프레임을 기다리는 중" 안내를 한 번만 찍기 위한 표시
    try:
        # 메인 루프. 한 번 도는 것이 한 프레임
        while True:
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
                # 기다리는 동안에도 Q / ESC 로 끝낼 수 있게 키를 확인한다
                if (cv2.waitKey(30) & 0xFF) in (ord("q"), 27):
                    break
                continue
            fails = 0
            # 논리 캔버스(1100x480) 생성. time.monotonic() = 현재 시각(초)으로 페이드, FPS, 가려짐 주기 계산에 쓰인다
            canvas = app.render(frame, time.monotonic())
            cv2.imshow(WIN_NAME, app.present(canvas))
            # 1ms 동안 키 입력 대기 (imshow 가 화면을 실제로 그리려면 waitKey 호출이 필요). & 0xFF 로 하위 8비트만 취해
            #   플랫폼 차이를 없앤다. 키가 없으면 255
            k = cv2.waitKey(1) & 0xFF
            if k != 255 and not app.key(k, time.monotonic()):
                break
    # 정상 종료든 예외든 반드시 실행: MediaPipe 해제, 카메라 반납, 모든 창 닫기
    finally:
        app.close()
        src.release()
        cv2.destroyAllWindows()
    return rc


# 이 파일을 직접 실행했을 때만 main() 을 호출 (다른 파일에서 import
#   하면 실행되지 않음 -> 테스트에서 import 해서 함수만 쓸 수 있는 이유)
if __name__ == "__main__":
    sys.exit(main())

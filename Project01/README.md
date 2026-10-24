# 01_project

가천대 온디바이스 AI 수업의 개인 프로젝트 폴더입니다. 아래 표에서 설명서를 클릭하면 각 프로젝트의 자세한 문서로 이동합니다.

## 프로젝트

| 프로젝트 | 한 줄 설명 | 실행 파일 | 설명서 |
|---|---|---|---|
| **AirBall Football** | 웹캠 앞에서 손으로 공을 잡아 던지고, 발·무릎·머리·손바닥으로 저글링하는 2D 게임 (MediaPipe 손 + 몸) | [`06_02_jetson.py`](06_02_jetson.py) | **[README_06_02_jetson.md](README_06_02_jetson.md)** |
| **Vision Quiz Lab** | 망가뜨린 카메라 영상을 5지선다 OpenCV 코드로 고쳐 보는 컴퓨터 비전 학습 퀴즈 (Jetson 버전) | [`06_03_jetson.py`](06_03_jetson.py) | **[README_06_03_jetson.md](README_06_03_jetson.md)** |

## 빠른 실행 (Jetson, 모니터가 연결된 터미널)

```bash
cd ~/Documents/git/vision-LLM-<학번>
if meidapipie 미설치시 -> Home/vision-llm 경로에서 시작 

# AirBall Football (저장소 루트에서 실행 권장, 모델 파일 2개 필요)
python3 01_project/06_02_jetson.py

# Vision Quiz Lab (옵션 없이 USB 카메라로 바로 실행)
cd 01_project
python3 06_03_jetson.py
```

## 폴더 안 파일

| 파일 | 설명 |
|---|---|
| `05.py` | AirBall 초기 프로토타입 (손 제스처로 공 잡기 / 던지기) |
| `06.py` | AirBall: 손 던지기 + 발·무릎 저글링 |
| `06_02.py` | AirBall: 손 던지기 + 발·무릎 저글링에 머리·손바닥 저글링, 디버그 패널, 녹화 / 재생을 더한 버전 |
| `06_02_jetson.py` | AirBall jetson 버전 |
| `06_03.py` | Vision Quiz Lab 원본 (Windows 웹캠 / Jetson 자동 선택) |
| `06_03_jetson.py` | Vision Quiz Lab Jetson 버전 (GStreamer USB 파이프라인 고정) |
| `readme_images/` | `README_06_03_jetson.md` 의 스크린샷 |

## 공통 준비물

| 항목 | 내용 |
|---|---|
| OpenCV | Jetson 에서는 **GStreamer 지원 빌드** (JetPack 기본 OpenCV 는 보통 켜져 있음) |
| MediaPipe 모델 | `src/models/MediaPipe/hand_landmarker.task` (두 프로젝트 공통), `pose_landmarker_full.task` (AirBall 만) |
| 카메라 | USB 카메라 `/dev/video0` (640×480, 30fps) |

모델 파일을 받는 명령은 [README_06_02_jetson.md 의 4장](README_06_02_jetson.md#4-모델-파일-받기)에 있습니다.

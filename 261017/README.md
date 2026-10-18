# 261017 — 02_Computer-vision.py (농구공 영역 검출)

`Vision-LLM-202136044` 저장소의 **261017 폴더** 과제 파일 `02_Computer-vision.py`에 대한 설명서입니다.
이 파일은 컴퓨터 비전 실습 중 **농구공 영역 검출의 마지막 단계**를 담고 있습니다. 색상 기반으로 만든 binary mask(`mask_enhanced`)에서 노이즈를 제거하고, 농구공 영역을 하나의 외곽선으로 추출해 원본 이미지 위에 시각화합니다.

---

## 1. 이 파일은 무엇인가

| 항목 | 내용 |
|------|------|
| 파일 | `261017/02_Computer-vision.py` |
| 목적 | 색상 마스크 → Morphological 연산 → 외곽선 검출 → 가장 큰 외곽선을 농구공으로 선택 → 시각화 |
| 담당 구간 | 과제의 `TODO` 3개 (Morphological 연산 / 외곽선 검출 / 최종 영역·외곽선 선택) + 결과 시각화 |
| 사용 이미지 | `src/images` 폴더의 농구공 이미지 (`basketball.jpg`, `basketball_crop.jpg` 등, 아래 3장 참고) |

> **참고**: 이 파일은 파이프라인의 **끝부분**입니다. `mask_enhanced`와 `basketball_enhanced_rgb`는 이 구간보다 앞선 단계(이미지 로드 → 보정 → 색상 마스크 생성)에서 만들어져 있어야 하며, 정의되어 있지 않으면 `NameError`가 발생합니다.

---

## 2. 저장소 구조

```text
C:\Users\minseok\Vision-LLM-202136044
├── 00_Initial-Setup.ipynb
├── 01_Linux-and-Python.ipynb
├── 02_Computer-Vision.ipynb
├── README.md                          # 저장소 전체 설명
│
├── 261010/
│   ├── 01_Linux-and-Python.py
│   └── README.md
│
├── 261017/                            # ← 이 문서가 있는 폴더
│   ├── 02_Computer-vision.py          # 이 README가 설명하는 과제 파일
│   └── README.md                      # 이 문서
│
└── src/
    ├── datasets/
    │   └── calibration/               # (비어 있음, .gitkeep)
    ├── images/                        # 실습 이미지
    │   ├── 1665_Girl_with_a_Pearl_Earring.jpg
    │   ├── apple.png
    │   ├── basketball.jpg
    │   ├── basketball_crop.jpg
    │   ├── binmask_crop.PNG
    │   ├── city.png
    │   ├── COCO_80_classes.png
    │   ├── seagull.jpg
    │   ├── sky.png
    │   ├── two_apples1.png
    │   ├── two_apples2.png
    │   └── two_apples3.png
    └── models/                        # (비어 있음, .gitkeep)
        ├── CIFAR10/
        ├── MediaPipe/
        ├── MNIST/
        └── YOLO/
```

### 중요한 경로 관계

`02_Computer-vision.py`는 `261017` 폴더 안에 있고, 이미지는 저장소 루트의 `src\images`에 있습니다. 즉 **스크립트 기준으로 한 단계 위(`..`)에 `src`가 있습니다.**

| 항목 | 절대 경로 |
|------|-----------|
| 저장소 루트 | `C:\Users\minseok\Vision-LLM-202136044` |
| 스크립트 | `C:\Users\minseok\Vision-LLM-202136044\261017\02_Computer-vision.py` |
| 이미지 폴더 | `C:\Users\minseok\Vision-LLM-202136044\src\images` |
| 스크립트 → 이미지 상대 경로 | `..\src\images` |

`src\datasets`와 `src\models` 아래 폴더들은 `.gitkeep`만 있는 빈 폴더라서 이 과제에는 필요하지 않습니다.

---

## 3. 환경 설정

### 3-1. 요구 사항

- Python 3.8 이상 (현재 터미널은 Anaconda `base` 환경)
- 라이브러리 3종

### 3-2. 사용 라이브러리

| 패키지 | 용도 | 코드 내 import |
|--------|------|----------------|
| `opencv-python` | 이미지 로드, Morphological 연산, 외곽선 검출, 오버레이 | `import cv2` |
| `numpy` | 배열 연산, 빈 mask 생성 (`np.zeros_like`) | `import numpy as np` |
| `matplotlib` | 결과 이미지 시각화 | `import matplotlib.pyplot as plt` |

> 패키지 이름은 `opencv-python`이지만 import 할 때는 `cv2`를 사용합니다.

### 3-3. 설치

PowerShell(Anaconda `(base)` 환경)에서 아래 중 하나로 설치합니다.

```bash
# pip
pip install opencv-python numpy matplotlib

# 또는 conda
conda install -c conda-forge opencv numpy matplotlib
```

설치 확인:

```bash
python -c "import cv2, numpy, matplotlib; print(cv2.__version__, numpy.__version__, matplotlib.__version__)"
```

버전이 출력되면 완료입니다. Anaconda에는 `numpy`, `matplotlib`이 이미 포함되어 있는 경우가 많고, 보통 `opencv-python`만 추가로 필요합니다.

---

## 4. 코드 구성

### 4-1. 파일 상단 import

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path   # 이미지 경로를 스크립트 위치 기준으로 만들기 위해 사용
```

### 4-2. 이미지 경로 설정

스크립트가 `261017` 안에 있으므로, 스크립트 위치 기준으로 한 단계 위의 `src\images`를 가리키도록 합니다. 이렇게 하면 어느 폴더에서 실행해도 동작합니다.

```python
BASE_DIR  = Path(__file__).resolve().parent.parent   # C:\Users\minseok\Vision-LLM-202136044
IMAGE_DIR = BASE_DIR / "src" / "images"              # ...\src\images

# 코드에서 실제로 사용하는 파일에 맞추세요 (basketball.jpg 또는 basketball_crop.jpg)
img_path = IMAGE_DIR / "basketball.jpg"

img_bgr = cv2.imread(str(img_path))
if img_bgr is None:
    raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {img_path}")

# OpenCV는 BGR로 읽으므로 matplotlib 출력용으로 RGB 변환
basketball_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
```

- `cv2.imread()`는 경로가 틀려도 예외 없이 `None`을 반환하므로 `None` 체크가 필요합니다.
- `__file__`은 `.py`로 실행할 때만 정의됩니다. Jupyter 노트북에서는 `Path.cwd()`를 기준으로 경로를 잡아야 합니다.
- 절대 경로를 직접 쓸 경우 raw string(`r"C:\Users\..."`) 또는 슬래시(`/`)를 사용합니다.

### 4-3. 선행 단계 (이 파일의 TODO 구간 이전)

아래 두 변수는 앞선 단계에서 만들어져 있어야 합니다.

| 변수 | 설명 | 조건 |
|------|------|------|
| `basketball_enhanced_rgb` | 보정된 RGB 이미지 | `H×W×3`, `uint8`, RGB 순서 |
| `mask_enhanced` | 색상 기반 binary mask | `H×W`, **`uint8`**, 값은 0 또는 255 |

> `cv2.findContours`는 **단일 채널 uint8** 이미지를 기대합니다. mask가 `bool`/`float`이면 `mask.astype(np.uint8) * 255`로 변환하세요.

---

## 5. 처리 흐름

| 단계 | 내용 | 사용 함수 |
|------|------|-----------|
| 1 | 타원형 구조 요소(5×5) 생성 | `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))` |
| 2 | Opening (2회) — 작은 노이즈 제거 | `cv2.morphologyEx(..., cv2.MORPH_OPEN, iterations=2)` |
| 3 | Closing (2회) — 내부 구멍 메우기 | `cv2.morphologyEx(..., cv2.MORPH_CLOSE, iterations=2)` |
| 4 | 최외곽 외곽선 검출 | `cv2.findContours(..., cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)` |
| 5 | 면적이 가장 큰 외곽선을 농구공으로 선택 | `max(contours, key=cv2.contourArea)` |
| 6 | 선택한 외곽선 내부를 흰색(255)으로 채운 mask 생성 | `np.zeros_like`, `cv2.drawContours(..., cv2.FILLED)` |
| 7 | 초록색 반투명 오버레이 + 초록색 외곽선 표시 | `cv2.addWeighted`, `cv2.drawContours` |
| 8 | 결과 3장 나란히 시각화 | `plt.subplot`, `plt.imshow` |

### 주요 파라미터

| 파라미터 | 값 | 의미 |
|----------|----|------|
| 구조 요소 | `MORPH_ELLIPSE`, `(5, 5)` | 공의 둥근 형태에 맞는 타원형 커널 |
| Opening / Closing 반복 | `iterations=2` | 값을 키우면 더 강하게 노이즈 제거 / 구멍 메우기 (공 크기가 줄거나 커질 수 있음) |
| 외곽선 모드 | `RETR_EXTERNAL` | 가장 바깥 외곽선만 검출 |
| 외곽선 근사 | `CHAIN_APPROX_SIMPLE` | 수평·수직·대각선 구간의 중복 점 제거 |
| 오버레이 투명도 | `0.35` | 초록색 영역이 원본에 섞이는 비율 |
| 외곽선 색 / 두께 | `(0, 255, 0)`, `2` | RGB 기준 초록색 |

---

## 6. 입력 / 출력

- **입력**
  - `mask_enhanced`: 이전 단계에서 생성한 binary mask
  - `basketball_enhanced_rgb`: 시각화에 사용할 RGB 이미지
- **출력**
  - `ball_mask_final`: Morphological 연산이 적용된 최종 binary mask
  - `ball_img`: 농구공 영역과 외곽선이 표시된 이미지

---

## 7. 실행

PowerShell에서 아래처럼 실행합니다. 스크립트 위치 기준으로 경로를 잡았다면 어느 폴더에서 실행해도 됩니다.

```bash
cd C:\Users\minseok\Vision-LLM-202136044\261017
python 02_Computer-vision.py
```

### 시각화 결과

`matplotlib`으로 세 장을 나란히 출력합니다.

1. **Enhanced Mask** — 입력 마스크
2. **Mask (open x2, close x2)** — Morphological 연산 후 마스크
3. **Largest Contour** — 가장 큰 외곽선을 표시한 결과

> **주의 (스크립트 실행 시)**: 현재 코드 끝에는 `plt.show()`가 없습니다. Jupyter Notebook에서는 그림이 자동으로 표시되지만, `python 02_Computer-vision.py`로 실행하면 창이 뜨지 않고 종료됩니다. 터미널 실행용이라면 마지막 줄에 다음을 추가하세요.
>
> ```python
> plt.show()
> ```

---

## 8. 문제 해결

| 증상 | 원인 | 해결 |
|------|------|------|
| `ModuleNotFoundError: No module named 'cv2'` | OpenCV 미설치 또는 다른 Python 환경에 설치됨 | 사용 중인 환경(`(base)` 등)에서 `pip install opencv-python` |
| `NameError: name 'mask_enhanced' is not defined` | 선행 단계 코드가 실행되지 않음 | 이미지 로드 ~ 마스크 생성 단계를 먼저 실행 |
| `cv2.imread`가 `None` 반환 | 경로·파일명·확장자 오류 | 스크립트가 `261017` 안에 있으므로 `..\src\images` 기준인지, 파일명 철자 확인 |
| `NameError: name '__file__' is not defined` | Jupyter 등 `.py`가 아닌 환경에서 실행 | `Path.cwd()` 기준으로 경로 설정 |
| `cv2.error` (`findContours` 관련) | mask의 dtype이 `uint8`이 아니거나 3채널 | `mask.astype(np.uint8)`, 단일 채널로 변환 |
| 실행해도 그림 창이 안 뜸 | `plt.show()` 누락 | 마지막 줄에 `plt.show()` 추가 |
| 색이 이상하게 보임 | BGR 이미지를 그대로 `imshow` | `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`로 변환 |
| 공이 아닌 영역이 선택됨 | 배경에 공보다 큰 비슷한 색 영역 존재 | 색상 범위 조정, ROI 제한, 면적·원형도 필터 추가 |

---

## 9. 참고

- Opening은 침식 후 팽창이라 작은 점 노이즈를 지우고, Closing은 팽창 후 침식이라 물체 내부의 빈 구멍을 메웁니다.
- 가장 큰 외곽선만 사용하므로, 배경에 농구공보다 큰 비슷한 색 영역이 있으면 잘못 선택될 수 있습니다.
- 초록색 오버레이는 RGB 이미지의 채널 인덱스 1(G)에만 mask를 넣어 만듭니다. 따라서 `ball_img`는 반드시 **RGB 순서**여야 의도한 색으로 표시됩니다.

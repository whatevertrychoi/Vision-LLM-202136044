# [261025] 과제 #3 — 딥러닝 GPU 가속 & 객체 탐지 기초

> **강의**: Physical AI의 Vision-LLM 융합 시청각 멀티모달 시스템 (가천대 × 한컴 AX LAB)
> **회차**: 3주차 6회차 (2026-10-25, 일)
> **대상 노트북**: `03_DL-and-GPU.ipynb`, `04_DL-Object-Detection.ipynb`
> **실행 환경**: NVIDIA Jetson Orin Nano (JetPack 6.2, CUDA 12.6) / 연습용 노트북 PC

---

## 목차

- [0. 과제 개요](#0-과제-개요)
- [폴더 구조](#폴더-구조)
- [실행 환경](#실행-환경)
- [공통 개념: GPU 시간 측정의 원리](#공통-개념-gpu-시간-측정의-원리)
- [1. GPU 가속 — 잔상 효과 & Convolution2D (`01_gpu_motion_trail_conv2d.py`)](#1-gpu-가속--잔상-효과--convolution2d)
- [2. MNIST `main_gpu()` (`02_mnist_main_gpu.py`)](#2-mnist-main_gpu)
- [3. CIFAR-10 학습 + 모델 저장 (`03_cifar10_train_save.py`)](#3-cifar-10-학습--모델-저장-pth)
- [4. IoU 구현 (`04_iou.py`)](#4-iou-구현)
- [5. NMS 구현 (`05_nms.py`)](#5-nms-구현)
- [제출 체크리스트](#제출-체크리스트)

---

## 0. 과제 개요

| # | 원본 노트북 | 과제 항목 | 노트북 내 위치 | 정리 파일 (임시명) |
|:-:|---|---|---|---|
| 1 | `03_DL-and-GPU.ipynb` | GPU 가속: 이미지 잔상 효과 + Convolution2D (NumPy vs CuPy) | G-4), G-5) | `01_gpu_motion_trail_conv2d.py` |
| 2 | `03_DL-and-GPU.ipynb` | MNIST 데이터셋 `main_gpu()` 함수 | I-3) | `02_mnist_main_gpu.py` |
| 3 | `03_DL-and-GPU.ipynb` | CIFAR-10 모델 학습 파트 + 저장한 모델 파일 (`.pth`) | J-11) ~ J-14) | `03_cifar10_train_save.py` |
| 4 | `04_DL-Object-Detection.ipynb` | IoU 구현 (좌표 변환 포함) | B | `04_iou.py` |
| 5 | `04_DL-Object-Detection.ipynb` | NMS 구현 (Confidence Thresholding 포함) | C | `05_nms.py` |

과제 1~3은 **"같은 연산을 CPU와 GPU에서 돌리면 무엇이 얼마나 빨라지고, 그 측정을 어떻게 공정하게 하는가"** 가 핵심이고, 과제 4~5는 **객체 탐지 후처리(post-processing)의 두 기둥인 IoU와 NMS** 를 직접 구현해 YOLO 내부 동작을 이해하는 것이 목표입니다.

---

## 폴더 구조

```text
Vision-LLM-202136044/                  # 레포 루트 (여기서 실행)
├── src/                               # 모든 날짜 폴더가 공유하는 리소스
│   ├── images/
│   │   ├── seagull.jpg                # 과제 1 입력 이미지
│   │   ├── two_apples1.png            # 과제 4 IoU 테스트 이미지
│   │   ├── two_apples2.png
│   │   └── two_apples3.png
│   ├── datasets/                      # torchvision이 자동 다운로드 (git에 올리지 않음)
│   │   ├── MNIST/
│   │   └── cifar-10-batches-py/
│   └── models/
│       ├── MNIST/
│       │   └── MNIST_CNN.pth          # SimpleCNN state_dict
│       └── CIFAR10/
│           └── CIFAR10_CNN.pth        # CIFAR10CNN 체크포인트(dict)
│
└── 261025/                            # 이번 과제 폴더
    ├── README.md                      # 본 문서
    ├── 01_gpu_motion_trail_conv2d.py  # 과제 1: 잔상 효과 + Conv2D (NumPy vs CuPy)
    ├── 02_mnist_main_gpu.py           # 과제 2: MNIST CPU/GPU 학습 속도 비교
    ├── 03_cifar10_train_save.py       # 과제 3: CIFAR-10 GPU 학습 + 체크포인트 저장/로드
    ├── 04_iou.py                      # 과제 4: BBox 좌표 변환 + IoU
    └── 05_nms.py                      # 과제 5: Confidence Thresholding + NMS (+ cv2 비교)
```

**설계 의도**
- 과제 하나 = 파일 하나. 각 파일은 노트북에서 해당 섹션만 떼어 단독 실행이 가능하도록 필요한 import, 상수, 함수 정의를 모두 포함합니다.
- `src/` 는 **레포 루트**에 하나만 둡니다. 이미지·데이터셋·모델을 날짜 폴더마다 복사하지 않고 공유하기 위함입니다. 특히 데이터셋은 수백 MB라 중복 다운로드를 피해야 합니다.
- 코드 속 경로는 노트북과 똑같은 `"src/..."` 상대 경로를 그대로 씁니다. 상대 경로는 **실행한 위치(현재 작업 디렉토리) 기준**이므로, 반드시 **레포 루트에서** 실행합니다. 이렇게 하면 노트북 코드를 수정 없이 옮길 수 있습니다.
- `src/datasets/` 는 용량이 커서 `.gitignore` 대상입니다. 첫 실행 시 `download=True` 로 자동 생성됩니다.
- `.pth` 파일은 데이터셋별로 `src/models/<데이터셋명>/` 에 분리 저장합니다. 모델 클래스가 다르므로(SimpleCNN vs CIFAR10CNN) 섞이지 않게 하기 위함입니다.

---

## 실행 환경

| 항목 | Jetson Orin Nano | 비고 |
|---|---|---|
| JetPack / CUDA | 6.2.1 / 12.6 | `dpkg -l \| grep nvidia-jetpack`, `cat /usr/local/cuda/version.json` |
| Python | 3.10 | 수업 커널 `Vision-LLM (Python3.10)` |
| NumPy | 1.21.5 → 1.23.0 | **2.x 금지** (GStreamer 연동 OpenCV와 충돌) |
| PyTorch / torchvision | 2.8.0 / 0.23.0 (cu126) | `pypi.jetson-ai-lab.io/jp6/cu126` |
| CuPy | `cupy-cuda12x==12.3.0` | `fastrlock==0.8.3` 선설치 |
| OpenCV | 4.x (GStreamer 빌드) | |

```bash
# 실행 예시 — 반드시 레포 루트에서 (src/ 상대 경로 기준)
cd ~/Vision-LLM-202136044
python 261025/01_gpu_motion_trail_conv2d.py
python 261025/02_mnist_main_gpu.py
python 261025/03_cifar10_train_save.py
python 261025/04_iou.py
python 261025/05_nms.py
```

> `261025/` 안에서 `python 01_...py` 로 실행하면 `src/images/seagull.jpg` 를 `261025/src/...` 에서 찾게 되어 `FileNotFoundError` 가 납니다. 폴더 안에서 실행하고 싶다면 코드의 경로를 전부 `"../src/..."` 로 바꿔야 합니다.

---

## 공통 개념: GPU 시간 측정의 원리

과제 1~3 모두 같은 측정 원칙을 따르므로 먼저 정리합니다.

### (1) GPU 호출은 비동기(asynchronous)다

CPU가 `cp.asarray(...)`, `model(images)` 같은 GPU 연산을 호출하면, CPU는 **GPU에 "이 작업 해줘"라고 명령을 큐에 넣고 바로 다음 줄로 넘어갑니다.** GPU가 실제로 계산을 끝냈는지는 기다리지 않습니다.

```text
CPU:  [호출A]─[호출B]─[time 측정!] ......................
GPU:        [  A 실행  ][     B 실행     ]
                              ↑ 이 시점엔 아직 B가 안 끝남
```

그래서 `synchronize()` 없이 종료 시각을 찍으면 **GPU가 일을 끝내기 전의 시간**을 재게 되어 측정값이 비현실적으로 작게 나옵니다.

| 라이브러리 | 동기화 함수 |
|---|---|
| CuPy | `cp.cuda.Stream.null.synchronize()` |
| PyTorch | `torch.cuda.synchronize()` |

**규칙**: 시작 시각을 찍기 **전에** 한 번(이전 작업 비우기), 종료 시각을 찍기 **전에** 한 번(현재 작업 완료 대기).

### (2) Warm-up

GPU의 첫 연산에는 CUDA Context 생성, 커널 컴파일(CuPy는 JIT), cuDNN 초기화, 메모리 풀 할당 같은 **1회성 비용**이 섞여 있습니다. 이걸 측정에 포함하면 GPU가 실제보다 느리게 보입니다. 그래서 **작은 더미 입력으로 같은 함수를 미리 한 번 돌려놓고** 측정합니다.

### (3) 측정 구간 분리

```text
GPU 총 비용 = Upload(Host→Device) + Compute + Download(Device→Host)
```

- **Compute-only speedup** = `cpu_time / gpu_time` → GPU 연산 자체의 성능
- **Total speedup** = `cpu_time / (upload + gpu + download)` → 실제 파이프라인에서 체감되는 성능

Jetson은 CPU와 GPU가 같은 물리 메모리를 쓰는 **통합 메모리(UMA)** 구조지만, `cp.asarray()` 는 여전히 별도 버퍼로 **복사**하므로 전송 비용이 0이 되지는 않습니다.

---

## 1. GPU 가속 — 잔상 효과 & Convolution2D

> 파일: `01_gpu_motion_trail_conv2d.py` · 노트북 위치: `03_DL-and-GPU.ipynb` → G. GPU 병렬 연산 (CuPy) → 4), 5)

### 1-1. 목표

같은 계산을 **"픽셀 하나씩 순차 처리(CPU/NumPy + Python 루프)"** 와 **"배열 전체를 한 번에 병렬 처리(GPU/CuPy)"** 로 구현해, 결과는 같고 속도만 다르다는 것을 확인합니다.

### 1-2. 원리

#### 잔상 효과 (Motion Trail)

원본 이미지를 오른쪽으로 `shift` 픽셀씩 민 복사본에 가중치 `w[shift]` 를 곱해 모두 더합니다.

$$
\text{output}(y, x) = \sum_{s=0}^{L-1} w_s \cdot \text{image}(y,\ x - s) \qquad (x - s \ge 0 \text{ 인 항만})
$$

- `w` 는 `linspace(1.0, 0.05, 32)` 후 합이 1이 되도록 정규화 → 가까운 잔상일수록 진하고, 전체 밝기는 원본과 비슷하게 유지됩니다.
- 1D 예시: `[10,20,30,40,50]`, `w=[0.6,0.3,0.1]`

  ```text
  shift 0: [10,20,30,40,50] × 0.6
  shift 1: [ 0,10,20,30,40] × 0.3
  shift 2: [ 0, 0,10,20,30] × 0.1
  ───────────────────────────────── (같은 위치끼리 합)
  ```

#### 두 구현의 반복 구조 차이 (핵심)

| | CPU (`motion_trail_np`) | GPU (`motion_trail_cp`) |
|---|---|---|
| 바깥 루프 | 출력 픽셀 `(y, x)` 전체 (720×1280) | `shift` (32회) |
| 안쪽 루프 | `shift` (32회) | **없음** — 배열 연산 한 줄 |
| Python 반복 횟수 | 약 2,950만 회 | **32회** |
| 픽셀 연산 | Python 인터프리터가 하나씩 | GPU 코어 1024개가 동시에 |

**덧셈의 교환법칙** 덕분에 "픽셀 먼저 → shift" 순서를 "shift 먼저 → 픽셀 전체" 로 바꿔도 결과가 같습니다. 이 루프 순서 교환이 GPU 병렬화의 본질입니다.

#### Convolution을 같은 방식으로 바꾸기

기존 `convolution2d()` 는 "출력 위치 하나 → 커널 크기 영역 crop → 가중합" 입니다. 이를 뒤집으면:

```text
커널 원소 (ky, kx) 하나 선택  (3×3이면 9회 반복)
→ 모든 출력 위치에서 그 원소와 곱해질 입력 픽셀들을 "슬라이싱 한 번"으로 선택
→ kernel[ky, kx] × (그 배열 전체)
→ output 전체에 누적
```

즉, **Convolution = (커널 원소 개수)만큼의 "이동된 이미지 × 상수" 누적** 이고, 잔상 효과는 커널이 `1 × L` 인 특수한 Convolution입니다. 두 실습이 한 쌍으로 묶인 이유입니다.

### 1-3. 코드와 작성 이유

#### (a) 잔상 효과 — CPU

```python
def motion_trail_np(image, weights):
    """
    NumPy를 이용해 이미지 잔상 효과를 생성한다.
    출력 픽셀을 Python 반복문으로 하나씩 계산한다.
    """
    height, width = image.shape
    trail_length = min(TRAIL_LENGTH, len(weights))

    output = np.zeros((height, width), dtype=np.float32)

    for output_y in range(height):
        for output_x in range(width):
            accumulated = 0.0

            for shift in range(trail_length):
                # 오른쪽으로 shift만큼 밀린 이미지 = 원본의 왼쪽 shift 위치 픽셀
                input_x = output_x - shift

                # 이미지 왼쪽 밖이면 그 이후 shift도 전부 밖 → 중단
                if input_x < 0:
                    break

                accumulated += weights[shift] * image[output_y, input_x]

            output[output_y, output_x] = accumulated

    return output
```

| 줄 | 이유 |
|---|---|
| `input_x = output_x - shift` | "이미지를 오른쪽으로 민다" = 출력 위치에서 보면 **왼쪽**의 원본 픽셀을 가져오는 것 |
| `if input_x < 0: break` | 범위 밖은 0으로 취급(zero padding). `shift`가 커질수록 `input_x`는 작아지므로 `continue`가 아니라 `break`로 불필요한 반복을 잘라냄 |
| `trail_length = min(TRAIL_LENGTH, len(weights))` | 워밍업 셀은 `weights_np[:4]` (길이 4)를 넘기는데, 루프를 `TRAIL_LENGTH(32)` 로 고정하면 `weights[4]` 에서 **IndexError**. 가중치 길이 기준으로 반복 |
| `dtype=np.float32` | 입력이 `/255.0` 정규화된 float32이므로 맞춰야 GPU 결과와 비교 시 오차가 최소 |

#### (b) 잔상 효과 — GPU

```python
def motion_trail_cp(image, weights):
    """
    CuPy를 이용해 이미지 잔상 효과를 생성한다.
    이동 거리는 Python에서 순차적으로 반복하지만,
    각 이동 거리의 이미지 전체 계산은 GPU에서 병렬 처리한다.
    """
    height, width = image.shape
    trail_length = min(TRAIL_LENGTH, int(weights.shape[0]), width)

    output = cp.zeros((height, width), dtype=cp.float32)

    for shift in range(trail_length):
        # 원본에서 오른쪽 끝 shift개 열을 뺀 영역 (밀려서 화면 밖으로 나가는 부분 제외)
        input_region = image[:, 0:width - shift]

        # 출력에서 왼쪽 shift개 열을 뺀 영역 (밀려서 비게 되는 부분 제외)
        output_region = output[:, shift:width]

        # 두 영역은 크기가 같음 → 원소별 곱 + 누적을 GPU가 병렬 처리
        output_region += weights[shift] * input_region

    return output
```

| 줄 | 이유 |
|---|---|
| `image[:, 0:width-shift]` / `output[:, shift:width]` | 두 슬라이스의 shape이 `(H, W-shift)` 로 동일 → 브로드캐스팅 없이 1:1 대응. 출력 `x` ↔ 입력 `x - shift` 관계를 **인덱스 계산 없이 슬라이싱 오프셋으로** 표현 |
| `output_region += ...` | 슬라이싱 결과는 **view** (메모리 공유). in-place 연산 `+=` 이 원본 `output` 을 직접 수정함. `output_region = output_region + ...` 로 쓰면 새 배열이 생겨 `output` 이 갱신되지 않음 |
| `min(..., width)` | 워밍업용 64×64 이미지처럼 폭이 작을 때 `width - shift` 가 음수가 되는 것 방지 |

#### (c) 업로드 / 워밍업 / 처리 / 다운로드

```python
# ── 업로드: NumPy(CPU 메모리) → CuPy(GPU 메모리) ──
upload_start = time.perf_counter()

input_cp = cp.asarray(input_np)      # 입력 이미지 H→D 복사
weights_cp = cp.asarray(weights_np)  # 가중치 H→D 복사

cp.cuda.Stream.null.synchronize()    # 복사 완료까지 대기
upload_end = time.perf_counter()
upload_time = upload_end - upload_start

# ── 워밍업: 작은 입력으로 커널 JIT 컴파일·메모리 풀 초기화 ──
warmup_image = cp.zeros((64, 64), dtype=cp.float32)
warmup_weights = cp.asarray(weights_np[:4])
_ = motion_trail_cp(image=warmup_image, weights=warmup_weights)
cp.cuda.Stream.null.synchronize()

# ── GPU 처리 ──
cp.cuda.Stream.null.synchronize()    # 이전 작업 비우기
gpu_start = time.perf_counter()
output_cp = motion_trail_cp(image=input_cp, weights=weights_cp)
cp.cuda.Stream.null.synchronize()    # 32번의 커널 실행이 모두 끝날 때까지 대기
gpu_end = time.perf_counter()
gpu_time = gpu_end - gpu_start

# ── 다운로드: CuPy → NumPy (matplotlib/OpenCV는 NumPy만 받음) ──
download_start = time.perf_counter()
output_cp_np = cp.asnumpy(output_cp) # D→H 복사 (동기 함수라 synchronize 불필요)
download_end = time.perf_counter()
download_time = download_end - download_start
```

- `cp.asnumpy()` 는 결과를 NumPy 배열로 **반환해야 하므로** 내부적으로 복사 완료까지 기다립니다. 그래서 다운로드 구간에는 별도 synchronize가 없어도 정확합니다.
- 결과 비교에서 `compute_speedup` 과 `total_speedup` 을 따로 출력하는 이유는 [공통 개념 (3)](#3-측정-구간-분리) 참고.

#### (d) Convolution — GPU 함수

```python
def convolution2d_cp(image, kernel, bias=0.0, stride=1, padding=0):
    """
    CuPy를 이용한 2차원 convolution.
    커널의 위치는 Python 반복문으로 순차 처리하고,
    각 위치에 해당하는 출력 배열 전체 연산은 GPU에서 처리한다.
    """
    kernel_height, kernel_width = kernel.shape

    # CPU 버전과 동일한 zero padding
    padded_image = cp.pad(
        image,
        ((padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0,
    )

    # 출력 크기 = floor((H + 2P - K) / S) + 1
    output_height = (padded_image.shape[0] - kernel_height) // stride + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1

    output = cp.zeros((output_height, output_width), dtype=cp.float32)

    # 커널 원소 개수(3x3 = 9)만큼만 Python 반복
    for ky in range(kernel_height):
        for kx in range(kernel_width):
            # 모든 출력 위치 (oy, ox)에서 kernel[ky, kx]와 곱해지는 입력 픽셀은
            # padded_image[oy*stride + ky, ox*stride + kx]
            # → 시작점 (ky, kx), 간격 stride 인 슬라이싱 한 번으로 전부 선택
            input_region = padded_image[
                ky : ky + stride * (output_height - 1) + 1 : stride,
                kx : kx + stride * (output_width - 1) + 1 : stride,
            ]

            output += kernel[ky, kx] * input_region

    # 편향은 모든 출력 픽셀에 한 번만 더함
    output += bias

    return output
```

| 줄 | 이유 |
|---|---|
| 슬라이스 끝 `ky + stride*(output_height-1) + 1` | 정확히 `output_height` 개의 원소만 선택되도록 계산한 끝 인덱스. `stride=1` 이면 `ky : ky + output_height` 와 같음 |
| `::stride` | stride > 1 일 때 건너뛰기를 슬라이싱 step으로 처리 → 함수 하나로 모든 stride 지원 |
| `output += bias` 를 루프 밖에 | CPU 버전은 출력 픽셀마다 `weighted_sum + bias` 한 번. 루프 안에 넣으면 9번 더해짐 |

#### (e) Convolution — CPU 처리

```python
# TODO: 기존에 정의한 convolution2d 함수를 사용하여 convolution을 CPU로 처리

cpu_start = time.perf_counter()

output_dx_np = convolution2d(
    image=input_np, kernel=dx_edge_filter, bias=dx_bias, stride=1, padding=1
)
output_dy_np = convolution2d(
    image=input_np, kernel=dy_edge_filter, bias=dy_bias, stride=1, padding=1
)

cpu_end = time.perf_counter()
cpu_time = cpu_end - cpu_start

# ReLU는 측정 구간 밖 → GPU 쪽도 같은 구간(Conv 2회)만 측정해야 공정
relu_dx_np = relu(output_dx_np)
relu_dy_np = relu(output_dy_np)
```

- `padding=1` + 3×3 커널 + `stride=1` → 출력 크기 = 입력 크기 (1280×720 유지). CPU/GPU 결과를 같은 크기로 비교하기 위함.
- `input_np` 는 잔상 효과에서 쓴 1280×720 갈매기 이미지를 **재사용** (TODO의 "변수 재사용 가능").

#### (f) Convolution — GPU 처리 (변환 → 워밍업 → 처리)

```python
# ── 1) 변환: 입력과 필터를 GPU로 ──
upload_start = time.perf_counter()
input_cp = cp.asarray(input_np)
dx_filter_cp = cp.asarray(dx_edge_filter)
dy_filter_cp = cp.asarray(dy_edge_filter)
cp.cuda.Stream.null.synchronize()
upload_time = time.perf_counter() - upload_start

# ── 2) 워밍업: cp.pad / 슬라이싱 / 곱셈 커널 미리 컴파일 ──
_ = convolution2d_cp(
    cp.zeros((64, 64), dtype=cp.float32), dx_filter_cp, bias=dx_bias, stride=1, padding=1
)
cp.cuda.Stream.null.synchronize()

# ── 3) 처리 ──
cp.cuda.Stream.null.synchronize()
gpu_start = time.perf_counter()

output_dx_cp = convolution2d_cp(input_cp, dx_filter_cp, bias=dx_bias, stride=1, padding=1)
output_dy_cp = convolution2d_cp(input_cp, dy_filter_cp, bias=dy_bias, stride=1, padding=1)

cp.cuda.Stream.null.synchronize()
gpu_time = time.perf_counter() - gpu_start

# ── 4) ReLU는 GPU에서 적용 후 CPU로 다운로드 ──
download_start = time.perf_counter()
relu_dx_cp_np = cp.asnumpy(cp.maximum(0, output_dx_cp))
relu_dy_cp_np = cp.asnumpy(cp.maximum(0, output_dy_cp))
download_time = time.perf_counter() - download_start
```

- 필터도 `cp.asarray` 로 올리는 이유: `kernel[ky, kx]` 가 GPU 스칼라가 되어 **CPU↔GPU 혼합 연산을 피함** (노트북 G-2의 "잘못된 연산 예시" 상황 방지).
- ReLU를 다운로드 전에 GPU에서 처리 → 이미 GPU에 있는 데이터를 CPU로 가져와서 처리하는 왕복을 없앰.

#### (g) CPU/GPU 결과 비교

```python
# TODO: CPU/GPU 결과 비교 및 출력

# 1) 정확성: 두 결과가 수치적으로 같은지
max_diff_dx = np.max(np.abs(relu_dx_np - relu_dx_cp_np))
max_diff_dy = np.max(np.abs(relu_dy_np - relu_dy_cp_np))
same_dx = np.allclose(relu_dx_np, relu_dx_cp_np, atol=1e-5)
same_dy = np.allclose(relu_dy_np, relu_dy_cp_np, atol=1e-5)

# 2) 속도
compute_speedup = cpu_time / gpu_time
gpu_total_time = upload_time + gpu_time + download_time
total_speedup = cpu_time / gpu_total_time

print("========== Correctness ==========")
print(f"Vertical   (dx) max |diff|: {max_diff_dx:.2e}  allclose: {same_dx}")
print(f"Horizontal (dy) max |diff|: {max_diff_dy:.2e}  allclose: {same_dy}")
print()
print("========== NumPy CPU ==========")
print(f"Processing time: {cpu_time:.4f} seconds")
print()
print("========== CuPy GPU ==========")
print(f"CPU to GPU: {upload_time:.4f} seconds")
print(f"GPU processing: {gpu_time:.4f} seconds")
print(f"GPU to CPU: {download_time:.4f} seconds")
print(f"Total GPU time: {gpu_total_time:.4f} seconds")
print()
print("========== Comparison ==========")
print(f"Compute-only speedup: {compute_speedup:.2f}x")
print(f"Including transfers: {total_speedup:.2f}x")
```

- `==` 대신 `np.allclose(atol=1e-5)` 를 쓰는 이유: float32 덧셈은 **결합법칙이 정확히 성립하지 않음**. CPU는 "픽셀마다 9개를 `np.sum`", GPU는 "9번에 걸쳐 누적" 이라 더하는 순서가 달라 마지막 자리에서 미세한 차이가 날 수 있습니다.
- 속도보다 **정확성 검증을 먼저** 출력: 결과가 다르면 속도 비교 자체가 의미 없기 때문입니다.

### 1-4. 실행 흐름

```text
seagull.jpg ─ imread(GRAY) ─ resize(1280×720) ─ /255.0 ─► input_np (float32)
   │
   ├─[CPU] motion_trail_np ─────────────────────────────► output_np
   └─[GPU] asarray ─► warm-up ─► motion_trail_cp ─► asnumpy ─► output_cp_np
                                                          │
                                       allclose 비교 + 시간 비교
   (같은 input_np 재사용)
   ├─[CPU] convolution2d ×2 ─► relu ───────────────────► relu_dx_np, relu_dy_np
   └─[GPU] asarray ─► warm-up ─► convolution2d_cp ×2 ─► maximum ─► asnumpy
```

### 1-5. 결과

| 항목 | 잔상 효과 | Convolution (dx + dy) |
|---|---:|---:|
| CPU 처리 시간 | `__ s` | `__ s` |
| CPU → GPU | `__ s` | `__ s` |
| GPU 처리 | `__ s` | `__ s` |
| GPU → CPU | `__ s` | `__ s` |
| Compute-only speedup | `__ x` | `__ x` |
| Including transfers | `__ x` | `__ x` |
| 결과 일치 (`allclose`) | `True / False` | `True / False` |

> CPU 잔상 효과는 Python 루프 약 2,950만 회라 노트북에서 수십 초, Jetson에서는 수 분 걸릴 수 있습니다. 정상입니다.

### 1-6. 주의점 / 시험 포인트

- **GPU가 빠른 진짜 이유는 "Python 루프 제거 + 병렬"** 입니다. 같은 NumPy라도 벡터화(슬라이싱 방식)하면 CPU에서도 수십 배 빨라집니다. 이 실습의 큰 속도 차이에는 "Python 루프 vs 벡터 연산" 효과가 상당 부분 섞여 있음을 알고 있어야 합니다.
- **`synchronize()` 위치**: 시작 전 + 종료 전. 빠지면 GPU 시간이 비현실적으로 작게(수 μs) 나옴.
- **view vs copy**: 기본 슬라이싱은 view → `+=` 가 원본에 반영. Fancy indexing(`arr[[0,1,2]]`)은 copy라 반영 안 됨.
- **CPU 배열과 GPU 배열은 직접 연산 불가**: `np_array + cp_array` → 에러. 반드시 `cp.asarray` / `cp.asnumpy` 로 한쪽에 맞춤.
- **출력 크기 공식**: $\lfloor (H + 2P - K)/S \rfloor + 1$

---

## 2. MNIST `main_gpu()`

> 파일: `02_mnist_main_gpu.py` · 노트북 위치: `03_DL-and-GPU.ipynb` → I. MNIST 데이터셋 GPU 가속 학습 → 3)

### 2-1. 목표

이미 제공된 `main_cpu()` 와 **완전히 같은 학습**을 GPU에서 수행하는 `main_gpu()` 를 작성하고, 두 함수가 반환한 총 학습 시간으로 GPU speedup을 계산합니다.

### 2-2. 원리

딥러닝 학습 1 step = 아래 6단계 (노트북 I-1 복습)

```text
1. 이미지·Label 준비 (→ device로 이동)
2. 순전파: logits = model(images)
3. Loss 계산: CrossEntropy(logits, labels)
4. Gradient 초기화: optimizer.zero_grad()
5. 역전파: loss.backward()
6. 가중치 수정: optimizer.step()
```

CPU 버전과 GPU 버전의 차이는 **딱 두 가지**뿐입니다.

| 차이 | CPU | GPU |
|---|---|---|
| device | `torch.device("cpu")` | `torch.device("cuda")` |
| 시간 측정 | `perf_counter()` 만 | `torch.cuda.synchronize()` + `perf_counter()` |

`train_one_epoch()` / `evaluate()` 는 내부에서 이미 `images.to(device)`, `labels.to(device)` 를 하도록 작성되어 있어 **수정 없이 재사용**합니다. "모델과 Tensor는 같은 device에 있어야 한다" 는 규칙(노트북 H-2-i)을 함수 인자 `device` 하나로 해결하는 설계입니다.

### 2-3. 코드와 작성 이유

```python
# TODO: GPU 기반 MNIST 데이터셋 학습 코드 작성

def main_gpu():
    # ── 1. 재현성: CPU 버전과 같은 시드 → 같은 초기 가중치·같은 셔플 순서 ──
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ── 2. 전처리: PIL(0~255, HxW) → Tensor(0.0~1.0, 1xHxW) ──
    transform = transforms.ToTensor()

    # ── 3. 데이터셋 (학습 6만 / 테스트 1만) ──
    train_dataset = datasets.MNIST(
        root=DATA_ROOT,
        train=True,
        transform=transform,
        download=True,
    )

    test_dataset = datasets.MNIST(
        root=DATA_ROOT,          # 원본 main_cpu는 "src/datasets" 하드코딩 → 상수로 통일
        train=False,
        transform=transform,
        download=True,
    )

    # ── 4. DataLoader ──
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,       # 64
        shuffle=True,                # 매 epoch 순서 섞기
        num_workers=NUM_WORKERS,     # 0 (Jetson 메모리 안정성)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,  # 1000 (역전파 없음 → 큰 배치 가능)
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # ── 5. GPU 지정 + 모델 이동 ──
    device = torch.device("cuda")

    model = SimpleCNN()
    model = model.to(device)         # 파라미터를 GPU 메모리로

    # ── 6. 손실 함수 & 옵티마이저 (CPU 버전과 동일) ──
    criterion = nn.CrossEntropyLoss()   # LogSoftmax + NLL → logits 그대로 입력
    optimizer = torch.optim.Adam(
        model.parameters(),          # 반드시 .to(device) 이후의 파라미터로 생성
        lr=LEARNING_RATE,
    )

    train_loss_history = []
    train_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []

    # ── 7. 전체 학습 시간 측정 시작 ──
    torch.cuda.synchronize()         # model.to(device) 등 남은 GPU 작업 비우기
    training_start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        epoch_start_time = time.perf_counter()   # 직전 epoch 끝에서 이미 sync 됨

        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )

        test_loss, test_accuracy = evaluate(
            model, test_loader, criterion, device,
        )

        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

        torch.cuda.synchronize()     # 이번 epoch의 모든 커널 완료 대기
        epoch_time = time.perf_counter() - epoch_start_time

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy * 100:.2f}% | "
            f"Test loss: {test_loss:.4f} | "
            f"Test accuracy: {test_accuracy * 100:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

    # ── 8. 전체 학습 시간 측정 종료 ──
    torch.cuda.synchronize()
    total_training_time = time.perf_counter() - training_start_time
    print(f"Total GPU training time: {total_training_time:.2f}s")

    return total_training_time       # 비교 셀에서 speedup 계산에 사용


if __name__ == "__main__":
    print("=== CPU Training ===")
    cpu_training_time = main_cpu()

    print("\n=== GPU Training ===")
    gpu_training_time = main_gpu()

    speedup = cpu_training_time / gpu_training_time
    time_difference = cpu_training_time - gpu_training_time

    print("\n=== CPU vs. GPU Comparison ===")
    print(f"CPU training time: {cpu_training_time:.2f}s")
    print(f"GPU training time: {gpu_training_time:.2f}s")
    print(f"Time difference: {time_difference:.2f}s")
    print(f"GPU speedup: {speedup:.2f}x")
```

**왜 이렇게 썼나**

| 결정 | 이유 |
|---|---|
| `main_cpu()` 의 구조를 그대로 복제 | 비교 실험은 **바꾸는 변수가 하나(device)** 여야 결과 차이를 device 탓으로 돌릴 수 있음 |
| 시드 3종 고정 | 초기 가중치, 셔플 순서가 같아야 CPU/GPU 정확도가 비슷하게 나옴 → "GPU라서 정확도가 달라졌다" 는 오해 방지 |
| `optimizer` 를 `model.to(device)` **뒤에** 생성 | optimizer는 생성 시점의 파라미터 텐서를 참조. 순서가 바뀌면 CPU 텐서를 업데이트하는 사고가 날 수 있음 |
| 시작 전 `synchronize()` | 앞선 `model.to(device)` 복사가 측정에 섞이지 않게 |
| epoch 시작 시 sync 생략 | 직전 epoch 끝에서 이미 sync 했으므로 GPU 큐가 비어 있음 |
| `return total_training_time` | `__main__` 블록에서 CPU 결과와 나눠 speedup 계산 |

### 2-4. 결과

| 항목 | CPU | GPU |
|---|---:|---:|
| Epoch 1 시간 | `__ s` | `__ s` |
| Epoch 2 시간 | `__ s` | `__ s` |
| Epoch 3 시간 | `__ s` | `__ s` |
| 최종 Test accuracy | `__ %` | `__ %` |
| **총 학습 시간** | `__ s` | `__ s` |
| **GPU speedup** | | `__ x` |

> 기대값: 3 epoch 후 테스트 정확도 약 98%. CPU/GPU 정확도는 거의 같아야 정상.

### 2-5. 주의점 / 시험 포인트

- **speedup이 생각보다 작을 수 있다**: SimpleCNN은 파라미터가 약 5만 개로 매우 작고, `num_workers=0` 이라 **데이터 로딩(CPU)이 병목**이 됩니다. GPU는 놀고 CPU가 배치를 만드느라 바쁜 상태. 모델이 크거나 배치가 클수록 GPU 이득이 커집니다 (→ 과제 3 CIFAR-10).
- **Epoch 1이 유독 느리다**: CUDA Context·cuDNN 초기화가 첫 epoch에 포함되기 때문. 과제 3처럼 학습 전 GPU Warm-up을 넣으면 해결됩니다.
- **`model.train()` / `model.eval()`**: `train_one_epoch` 와 `evaluate` 안에서 각각 호출. Dropout·BatchNorm이 있는 모델에서 필수 (SimpleCNN엔 없지만 습관적으로).
- **`torch.no_grad()`**: 평가 시 계산 그래프를 만들지 않아 메모리·시간 절약.
- `CrossEntropyLoss` 에는 **softmax 하지 않은 logits** 를 넣는다 (내부에 LogSoftmax 포함).

---

## 3. CIFAR-10 학습 + 모델 저장 (.pth)

> 파일: `03_cifar10_train_save.py` · 노트북 위치: `03_DL-and-GPU.ipynb` → J. CIFAR-10 → 11) ~ 14)
> 산출물: `src/models/CIFAR10/CIFAR10_CNN.pth`

### 3-1. 목표

MNIST보다 어려운 CIFAR-10(컬러 32×32, 10클래스, 복잡한 배경)을 더 깊은 CNN으로 GPU에서 5 epoch 학습하고, **학습 결과를 체크포인트 파일로 저장 → 다시 불러와 추론**까지 확인합니다.

### 3-2. 원리

#### MNIST 대비 달라진 점

| 항목 | MNIST | CIFAR-10 | 이유 |
|---|---|---|---|
| 입력 | 1×28×28 흑백 | 3×32×32 RGB | |
| 전처리 | `ToTensor` | + `Normalize`, 학습 시 `RandomCrop`, `RandomHorizontalFlip` | 채널별 분포 정규화로 학습 안정, 증강으로 과적합 완화 |
| 모델 | Conv 2층 (SimpleCNN) | ConvBlock 3개 (Conv 6층) + BN + GAP + Dropout | 복잡한 특징 추출 |
| Optimizer | Adam | AdamW | Weight decay를 올바르게 분리 적용 |
| 배치 | 64 / 1000 | 32 / 128 | 모델·입력이 커서 Jetson 메모리 고려 |
| Warm-up | 없음 | 있음 | 첫 epoch 시간 왜곡 제거 |
| 저장 | `state_dict` 만 | 체크포인트 dict (모델+optimizer+메타데이터) | 재현·재학습·추론에 필요한 정보 동봉 |

#### CIFAR10CNN 구조 (요약)

```text
[B, 3, 32, 32]
 → ConvBlock(3→64)    → [B,  64, 16, 16]
 → ConvBlock(64→128)  → [B, 128,  8,  8]
 → ConvBlock(128→256) → [B, 256,  4,  4]
 → AdaptiveAvgPool2d  → [B, 256,  1,  1]
 → Flatten → Dropout(0.3) → Linear(256→10) → logits [B, 10]

ConvBlock = (Conv3x3 → BN → ReLU) × 2 → MaxPool2x2
```

### 3-3. 코드와 작성 이유

#### (a) 학습 루프 — 원래 코드의 버그 4개와 수정

| # | 원래 코드 | 문제 | 수정 |
|:-:|---|---|---|
| 1 | `train_one_epoch(model, test_loader, ...)` | **테스트셋으로 학습** → 데이터 누수. 테스트 정확도가 비정상적으로 높게 나와 평가 의미 상실 | `train_loader` |
| 2 | `train_loss, test_accuracy = evaluate(...)` | 학습 loss를 테스트 loss로 덮어씀 | `test_loss, test_accuracy` |
| 3 | `test_loss_history.append(test_loss)` | 2번 때문에 `test_loss` 미정의 → `NameError` | 2번 수정으로 해결 |
| 4 | `epoch_training_time = ... - epoch_start_time` 후 `total_training_time` 출력 | 마지막 epoch 시작 기준으로 측정 + 변수명 불일치 → `NameError` | `training_start_time` 기준, `total_training_time` 으로 통일 |

```python
# TODO: CUDA GPU 장치에서 CIFAR-10 데이터셋을 학습과 평가
# TODO: 각 Epoch 및 전체 학습 시간을 출력

EPOCHS = 5

# 학습 기록 (epoch별 → 학습 곡선 그래프용)
train_loss_history = []
train_accuracy_history = []
test_loss_history = []
test_accuracy_history = []

# 전체 학습 시간 측정 시작 (Warm-up 등 이전 GPU 작업 비우기)
torch.cuda.synchronize()
training_start_time = time.perf_counter()

for epoch in range(EPOCHS):

    torch.cuda.synchronize()
    epoch_start_time = time.perf_counter()

    # 학습: 50,000장, 데이터 증강 적용된 train_loader
    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
    )

    # 평가: 10,000장, 증강 없는 test_loader, 가중치 업데이트 X
    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    train_loss_history.append(train_loss)
    train_accuracy_history.append(train_accuracy)
    test_loss_history.append(test_loss)
    test_accuracy_history.append(test_accuracy)

    torch.cuda.synchronize()
    epoch_time = time.perf_counter() - epoch_start_time

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train accuracy: {train_accuracy * 100:.2f}% | "
        f"Test loss: {test_loss:.4f} | "
        f"Test accuracy: {test_accuracy * 100:.2f}% | "
        f"Time: {epoch_time:.2f}s"
    )

# TODO: 전체 학습 시간 계산 및 출력
torch.cuda.synchronize()
total_training_time = time.perf_counter() - training_start_time
print(f"Total GPU training time: {total_training_time:.2f}s")
```

- `train_one_epoch()` / `evaluate()` 는 MNIST에서 정의한 것을 **그대로 재사용**합니다. 함수가 모델·로더·device를 인자로 받도록 일반화되어 있어 데이터셋이 바뀌어도 수정이 필요 없습니다.
- 학습 전에 **GPU Warm-up** 셀(J-11, 더미 입력 3회 forward)이 실행되어 있어야 epoch 1 시간이 왜곡되지 않습니다.

#### (b) 모델 저장 — 체크포인트 dict

```python
save_data = {
    "model_name": "CIFAR10CNN",                       # 어떤 클래스로 복원할지
    "model_state_dict": model.state_dict(),           # 학습된 가중치 (핵심)
    "optimizer_state_dict": optimizer.state_dict(),   # AdamW 모멘트 → 이어서 학습 가능
    "test_accuracy": test_accuracy_history[-1],       # 이 체크포인트의 성능 기록
    "epochs": EPOCHS,                                 # 몇 epoch 학습했는지
    "class_names": CIFAR10_CLASSES,                   # 추론 결과 index → 이름 변환용
    "mean": CIFAR10_MEAN,                             # 추론 시 동일 정규화에 필요
    "std": CIFAR10_STD,
}

torch.save(save_data, "src/models/CIFAR10/CIFAR10_CNN.pth")
```

**왜 `model` 객체 통째가 아니라 `state_dict` 인가**
- `torch.save(model)` 은 pickle로 **클래스 정의 경로까지** 저장 → 코드 구조가 바뀌면 로드 실패.
- `state_dict` 는 `{레이어 이름: 텐서}` 딕셔너리일 뿐이라, 같은 구조의 클래스만 있으면 어디서든 로드 가능. PyTorch 공식 권장 방식.

**왜 MNIST처럼 `state_dict` 만 저장하지 않고 dict로 감쌌나**
- 추론 쪽에서 **정규화 mean/std와 클래스 이름이 반드시 필요**한데, 이를 코드에 따로 하드코딩하면 학습 때와 어긋날 위험이 있음 → 가중치와 함께 묶어 "이 파일 하나면 추론 가능" 하게 만듦.
- `optimizer_state_dict` 가 있으면 5 epoch 이후부터 **이어서 학습(resume)** 가능.

#### (c) 모델 불러오기

```python
loaded_model = CIFAR10CNN()                   # 1) 같은 구조의 빈 모델 생성

state_dict = torch.load(
    "src/models/CIFAR10/CIFAR10_CNN.pth",
    map_location="cpu",                       # 2) 어떤 장치에서 저장했든 일단 CPU로 로드
    weights_only=True,                        # 3) 텐서·기본 타입만 허용 (임의 코드 실행 방지)
)

loaded_model.load_state_dict(state_dict["model_state_dict"])  # 4) dict 안의 가중치만 꺼내 주입
loaded_model.to(device)                       # 5) 추론할 장치로 이동
loaded_model.eval()                           # 6) Dropout OFF, BN은 학습된 running 통계 사용
```

| 단계 | 이유 |
|---|---|
| `map_location="cpu"` | GPU에서 저장한 텐서를 GPU 없는 환경(노트북)에서도 열 수 있게. 이후 `.to(device)` 로 원하는 장치로 |
| `weights_only=True` | `.pth` 는 pickle 기반이라 악성 파일이면 로드 시 코드 실행 가능 → 텐서/기본 타입만 허용하는 안전 모드 |
| `["model_state_dict"]` | 체크포인트가 dict라서 **키로 꺼내야 함**. MNIST처럼 바로 `load_state_dict(state_dict)` 하면 키 불일치 에러 |
| `.eval()` | 이걸 빼면 Dropout이 켜진 채 추론 → 같은 이미지도 매번 다른 결과, BN이 배치 통계를 써서 1장 추론 시 결과가 망가짐 |

### 3-4. 실행 흐름

```text
CIFAR-10 다운로드 ─► train_transform(Crop+Flip+Normalize) / test_transform(Normalize)
   ─► DataLoader(32 / 128, pin_memory=True)
   ─► CIFAR10CNN().to("cuda") ─► CrossEntropy + AdamW(lr=1e-3)
   ─► GPU Warm-up (zeros[64,3,32,32] × 3회)
   ─► 5 epoch 학습/평가 (epoch별 시간 + 총 시간)
   ─► Loss / Accuracy 그래프
   ─► torch.save(checkpoint dict) ─► src/models/CIFAR10/CIFAR10_CNN.pth
   ─► torch.load ─► load_state_dict ─► eval ─► test_dataset[40:80] 추론 시각화
```

### 3-5. 결과

| Epoch | Train loss | Train acc | Test loss | Test acc | Time |
|:-:|---:|---:|---:|---:|---:|
| 1 | `__` | `__ %` | `__` | `__ %` | `__ s` |
| 2 | `__` | `__ %` | `__` | `__ %` | `__ s` |
| 3 | `__` | `__ %` | `__` | `__ %` | `__ s` |
| 4 | `__` | `__ %` | `__` | `__ %` | `__ s` |
| 5 | `__` | `__ %` | `__` | `__ %` | `__ s` |
| **Total** | | | | | **`__ s`** |

- 저장 파일: `src/models/CIFAR10/CIFAR10_CNN.pth` (`__ MB`)
- 로드 후 출력: `모델 이름: CIFAR10CNN`, `Epoch 체크포인트: 5`, `마지막 테스트 정확도: __ %`

> 5 epoch이면 테스트 정확도 약 70~80%대가 일반적입니다. 학습 정확도보다 테스트 정확도가 **비정상적으로 높다면** 버그 1번(test_loader로 학습)을 의심하세요.

### 3-6. 주의점 / 시험 포인트

- **데이터 증강은 학습 데이터에만**, **정규화는 학습·테스트 모두 동일하게**. 테스트에 증강을 넣으면 평가가 매번 달라지고, 정규화를 빼면 입력 분포가 학습 때와 달라 정확도가 폭락합니다.
- **역정규화 시각화**: `image * std + mean` 후 `.permute(1, 2, 0)` (C,H,W → H,W,C) + `.clamp(0, 1)`. 정규화된 값을 그대로 `imshow` 하면 "Clipping input data" 경고.
- **`pin_memory=True`**: 페이지 잠금 메모리를 써서 CPU→GPU 전송을 빠르게 함 (GPU 학습에서만 의미).
- **BatchNorm이 있는 Conv는 `bias=False`**: BN이 평균을 빼고 자체 β(shift)를 더하므로 Conv의 bias는 상쇄되어 무의미 → 파라미터 절약.
- **GAP(AdaptiveAvgPool2d)**: 256×4×4 를 Flatten하면 4096차원이지만, GAP 후 256차원 → FC 파라미터 16배 감소, 과적합 완화.
- **저장/로드 짝 맞추기**: 저장을 dict로 했으면 로드도 `["model_state_dict"]` 로 꺼낸다.

---

## 4. IoU 구현

> 파일: `04_iou.py` · 노트북 위치: `04_DL-Object-Detection.ipynb` → B. Intersection over Union

### 4-1. 목표

두 Bounding Box가 얼마나 겹치는지를 0~1 사이 값으로 계산하는 IoU 함수를 직접 구현하고, 색상 분할로 얻은 빨간 사과·초록 사과 BBox 3쌍에 적용합니다.

### 4-2. 원리

$$
\text{IoU} = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}
$$

- 분모에서 교집합을 한 번 빼는 이유: $|A| + |B|$ 에는 겹친 부분이 **두 번** 들어가 있기 때문.
- 0 = 전혀 안 겹침, 1 = 완전히 동일.

#### 교집합 사각형 좌표 (OpenCV 좌표계: x→오른쪽, y→아래)

```text
       xa1         xb1        xa2         xb2
  ya1  ┌───────────────────────┐
       │ A                     │
  yb1  │           ┌───────────┼───────────┐
       │           │ ▓▓▓▓▓▓▓▓▓ │           │
       │           │ ▓ 교집합 ▓ │         B │
  ya2  └───────────┼───────────┘           │
                   │                       │
  yb2              └───────────────────────┘

  교집합 좌상단 = (max(xa1, xb1), max(ya1, yb1))   ← 더 안쪽(오른쪽·아래)
  교집합 우하단 = (min(xa2, xb2), min(ya2, yb2))   ← 더 안쪽(왼쪽·위)
```

### 4-3. 코드와 작성 이유

```python
# TODO: 1) BBox 좌표를 (x,y,w,h)에서 (x1,y1,x2,y2)로 변환하는 함수 정의
# TODO: 2) IoU 계산 함수 정의

def convert_bbox_coord(bbox):
    """
    (x, y, w, h) → (x1, y1, x2, y2)
    (x, y)는 좌상단. 이미지 좌표계는 y가 아래로 증가하므로 y + h 가 아래쪽 경계.
    """
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def calculate_iou(bbox_a, bbox_b):
    """
    IoU = 교집합 면적 / 합집합 면적  (입력은 (x1, y1, x2, y2) 형식)
    """
    xa_1, ya_1, xa_2, ya_2 = bbox_a
    xb_1, yb_1, xb_2, yb_2 = bbox_b

    # 1. 교집합 사각형 좌표
    intersection_x1 = max(xa_1, xb_1)
    intersection_y1 = max(ya_1, yb_1)
    intersection_x2 = min(xa_2, xb_2)
    intersection_y2 = min(ya_2, yb_2)

    # 2. 교집합 면적 (안 겹치면 음수 → 0으로 클리핑)
    w = max(0, intersection_x2 - intersection_x1)
    h = max(0, intersection_y2 - intersection_y1)
    intersection_area = w * h

    # 3. 각 박스 면적
    area_a = (xa_2 - xa_1) * (ya_2 - ya_1)
    area_b = (xb_2 - xb_1) * (yb_2 - yb_1)

    # 4. 합집합 = A + B - 교집합
    union_area = area_a + area_b - intersection_area

    # 5. 0으로 나누기 방지 (두 박스 모두 면적 0인 퇴화 박스)
    if union_area <= 0:
        return 0.0

    # TODO: ZeroDivisionError 주의 → 위 조건문으로 처리
    return intersection_area / union_area
```

| 결정 | 이유 |
|---|---|
| 변환 함수를 따로 분리 | `cv2.boundingRect()` 는 `(x, y, w, h)` 를 반환하지만 IoU 계산은 `(x1, y1, x2, y2)` 가 편함. 형식 변환과 계산을 분리해 IoU 함수는 **한 가지 형식만** 받도록 단순화 |
| `max(0, ...)` 를 w, h **각각에** | 두 박스가 x, y 양쪽으로 어긋나면 w, h 가 **둘 다 음수** → 곱하면 양수가 되어 "안 겹치는데 IoU > 0" 오류. 각각 클리핑해야 함 |
| `union_area <= 0` 검사 | 노트북 TODO의 ZeroDivisionError 대응. `== 0` 이 아니라 `<= 0` 으로 비정상 좌표(x2 < x1)까지 방어 |
| `max/min` 내장 함수 사용 | 박스 2개끼리의 비교라 NumPy가 필요 없음. 다수 박스 일괄 계산이면 `np.maximum/np.minimum` 으로 벡터화 |

#### 사용 예 (노트북 셀)

```python
bbox1 = cv2.boundingRect(cntr_red1)   # (x, y, w, h)
bbox2 = cv2.boundingRect(cntr_grn1)

iou1 = calculate_iou(convert_bbox_coord(bbox1), convert_bbox_coord(bbox2))
print(iou1)
```

### 4-4. 결과

| 이미지 | 두 사과 배치 | IoU |
|---|---|---:|
| `two_apples1.png` | | `__` |
| `two_apples2.png` | | `__` |
| `two_apples3.png` | | `__` |

> 직접 검산: `A=(0,0,10,10)`, `B=(5,5,15,15)` → 교집합 25, 합집합 175 → IoU = 0.1429

### 4-5. 주의점 / 시험 포인트

- **교집합은 `max` 로 시작, `min` 으로 끝.** 헷갈리면 "교집합은 두 박스 어느 쪽보다도 작다" 로 기억.
- **BBox 표현 4가지**를 구분할 것:

  | 형식 | 의미 | 사용처 |
  |---|---|---|
  | `(x1, y1, x2, y2)` | 좌상단·우하단 | `cv2.rectangle`, YOLO `boxes.xyxy` |
  | `(x, y, w, h)` | 좌상단 + 크기 | `cv2.boundingRect`, `cv2.dnn.NMSBoxes` |
  | `(cx, cy, w, h)` | 중심 + 크기 | YOLO 학습 라벨 |
  | `(cx, cy, w, h)` / 이미지 크기 | 0~1 정규화 | YOLO `.txt` 라벨 파일 |

- 중심 좌표 형식의 변환은 `(cx − w/2, cy − h/2, cx + w/2, cy + h/2)` 로 **식이 다르다.**
- IoU 활용처: ① NMS에서 중복 판단, ② mAP 계산에서 TP 판정 (mAP@0.5 = IoU ≥ 0.5면 정답).

---

## 5. NMS 구현

> 파일: `05_nms.py` · 노트북 위치: `04_DL-Object-Detection.ipynb` → C. Non-Maximum Suppression

### 5-1. 목표

랜덤으로 생성한 20개의 겹치는 BBox에서 **Confidence Thresholding → NMS** 를 직접 구현해 중복을 제거하고, OpenCV 내장 `cv2.dnn.NMSBoxes()` 결과와 일치하는지 확인합니다.

### 5-2. 원리

객체 탐지 모델은 같은 객체 주변에 **조금씩 다른 박스를 여러 개** 예측합니다. NMS는 그중 가장 확신도 높은 하나만 남기는 후처리입니다.

```text
1. Confidence < conf_threshold 인 박스 제거           ← prune_low_conf
2. 남은 박스를 score 내림차순 정렬
3. 1등 박스를 확정(keep)
4. 1등과 IoU > iou_threshold 인 박스 제거 (같은 객체로 판단)
5. 남은 후보 중 다시 1등 선택 → 3~5 반복, 후보가 없으면 종료   ← apply_nms
```

### 5-3. 코드와 작성 이유

#### (a) Confidence Thresholding

```python
# TODO: Confidence Thresholding 함수 정의
# TODO: Confidence Score가 특정 임계값보다 낮으면 해당 BBox 제거

def prune_low_conf(boxes, scores, conf_threshold=0.5):
    """
    Confidence Thresholding 적용

    Returns
    -------
    pruned_boxes : list   Confidence Score가 임계값 이상인 Bounding Box 목록
    pruned_scores : list  그 박스들의 Confidence Score 목록
    """
    pruned_boxes = []
    pruned_scores = []

    # 박스와 점수를 같은 인덱스끼리 묶어서 순회 → 짝이 어긋날 일이 없음
    for box, score in zip(boxes, scores):
        if score >= conf_threshold:
            pruned_boxes.append(box)
            pruned_scores.append(score)

    return pruned_boxes, pruned_scores
```

- **NMS보다 먼저** 하는 이유: NMS는 O(N²) 이라 후보를 미리 줄이면 계산량이 크게 감소. 또한 낮은 점수 박스가 "1등"으로 뽑혀 다른 박스를 지우는 일을 막음.
- `zip` 으로 묶어서 처리: `boxes` 와 `scores` 를 따로 필터링하면 인덱스가 어긋날 위험.

#### (b) NMS

```python
# TODO: NMS 함수 정의
# TODO: np.sort() 함수를 사용하면 1차원 배열을 정렬
# TODO: np.argsort() 함수를 사용하면 1차원 배열의 정렬된 인덱스를 반환

def apply_nms(boxes, scores, iou_threshold=0.5):
    """
    NMS 적용

    Returns
    -------
    nms_boxes : list   NMS 적용 후 유지된 Bounding Box 목록
    nms_scores : list  그 박스들의 Confidence Score 목록
    """
    # 입력 박스가 없으면 바로 빈 결과 반환
    if len(boxes) == 0:
        return [], []

    # score 내림차순 인덱스
    # argsort: 오름차순 "인덱스" → [::-1] 로 뒤집어 내림차순 → pop/컴프리헨션 위해 list로
    # 예) scores=[0.7, 0.9, 0.6] → argsort=[2,0,1] → 뒤집기=[1,0,2]
    order = np.argsort(scores)[::-1].tolist()

    keep = []

    while order:
        # 남은 후보 중 최고 점수 박스를 꺼내 확정
        best = order.pop(0)
        keep.append(best)

        # best와 많이 겹치는(IoU > threshold) 박스 제거, 나머지만 다음 라운드 후보로
        order = [
            i for i in order
            if calculate_iou(boxes[best], boxes[i]) <= iou_threshold
        ]

    # keep은 이미 score 내림차순
    nms_boxes = [boxes[i] for i in keep]
    nms_scores = [scores[i] for i in keep]

    return nms_boxes, nms_scores
```

| 결정 | 이유 |
|---|---|
| `np.argsort` (값이 아니라 **인덱스** 정렬) | `np.sort(scores)` 는 점수만 정렬되어 어느 박스의 점수인지 잃어버림. 인덱스를 정렬해야 `boxes[i]`, `scores[i]` 를 함께 추적 가능 |
| `[::-1]` | argsort는 오름차순 → 높은 점수부터 처리하려면 뒤집기 |
| `.tolist()` | NumPy 배열엔 `pop` 이 없음. 파이썬 리스트로 바꿔 `pop(0)` 과 리스트 컴프리헨션 사용 |
| 남기는 조건 `<=` | 제거 조건이 `IoU > threshold` 이므로 그 반대. `cv2.dnn.NMSBoxes` 와 같은 기준이라 결과 일치 |
| 앞서 만든 `calculate_iou` 재사용 | 과제 4의 함수를 그대로 사용. 박스 형식이 둘 다 `(x1, y1, x2, y2)` 라 변환 불필요 |
| 원래 skeleton의 마지막 `pass` 삭제 | `return` 뒤라 실행되지 않는 dead code |

#### (c) 적용 셀 — 원래 코드의 연결 버그와 수정

```python
print(len(boxes), len(scores))

new_boxes, new_scores = prune_low_conf(boxes, scores)
new_boxes, new_scores = apply_nms(new_boxes, new_scores, iou_threshold=0.2)

print(len(new_boxes), len(new_scores))
```

```python
hubble = skimage.data.hubble_deep_field()

for i, box in enumerate(new_boxes):
    draw_bbox(
        hubble,
        (box[0], box[1]),
        (box[2], box[3]),
        (randint(0, 255), randint(0, 255), randint(0, 255)),
        txt=f"{new_scores[i]:.2f}",
    )
```

| 원래 코드 | 문제 |
|---|---|
| `apply_nms(boxes, scores, ...)` | `prune_low_conf` 결과를 쓰지 않음. 반면 `apply_nms_cv2` 는 내부에서 `conf_threshold=0.5` 를 적용 → **직접 구현과 cv2 결과가 달라지는 원인** |
| `txt=f"{scores[i]:.2f}"` | `new_boxes` 의 i번째와 원본 `scores` 의 i번째는 다른 박스. 화면에 표시되는 점수가 틀림 (cv2 결과 그리는 셀도 동일 → `cv2_new_scores[i]`) |

#### (d) OpenCV 내장 NMS와 비교

```python
cv2_new_boxes, cv2_new_scores = apply_nms_cv2(boxes, scores, iou_threshold=0.2)
# 내부에서 (x1,y1,x2,y2) → (x,y,w,h) 변환 후 cv2.dnn.NMSBoxes(score_threshold=0.5, nms_threshold=0.2)

# 검증: 직접 구현 결과와 동일한 박스 집합인지
print(sorted(new_boxes) == sorted(cv2_new_boxes))
```

- `apply_nms_cv2` 가 `(x, y, w, h)` 로 변환하는 이유: `cv2.dnn.NMSBoxes` 는 `cv2.Rect` 형식(좌상단+크기)만 받음. 과제 4의 BBox 형식 구분이 여기서 다시 쓰임.
- 두 결과가 같으려면 **조건이 같아야** 합니다: conf 0.5 → prune, IoU 0.2 → NMS. (c)의 수정이 이를 맞추는 작업입니다.
- 극히 드물게, score가 정확히 0.5인 박스는 다를 수 있습니다 (직접 구현은 `>=`, OpenCV는 `>`).

### 5-4. 실행 흐름

```text
hubble_deep_field 이미지
 └─ random_bbox × 20, random_conf(가우시안 μ=0.6, σ=0.1) × 20
     ├─[직접 구현] prune_low_conf(0.5) ─► apply_nms(IoU 0.2) ─► 그리기
     └─[OpenCV]   apply_nms_cv2(conf 0.5, IoU 0.2)          ─► 그리기
                                    └─ 좌우 비교 + 박스 집합 일치 확인
```

### 5-5. 결과

| 단계 | 박스 수 |
|---|---:|
| 생성 | 20 |
| Confidence Thresholding 후 (≥ 0.5) | `__` |
| 직접 구현 NMS 후 (IoU 0.2) | `__` |
| OpenCV NMS 후 | `__` |
| 두 결과 일치 여부 | `True / False` |

> 랜덤 생성이라 실행마다 개수가 다릅니다. 재현하려면 상단에 `random.seed(42)` 추가.

### 5-6. 주의점 / 시험 포인트

- **한 줄 요약**: "점수 최고 박스 확정 → 그 박스와 IoU가 임계값 넘는 박스 제거 → 반복"
- **IoU 임계값 방향**:
  - 낮추면 (예 0.2) → 조금만 겹쳐도 제거 → 붙어 있는 **다른 객체까지 지워질** 위험
  - 높이면 (예 0.7) → 많이 겹쳐야 제거 → **중복 박스가 남을** 위험
  - YOLO 기본값은 보통 0.45~0.7
- **Confidence vs IoU 임계값** 구분: conf = "객체가 맞나", IoU = "같은 객체인가"
- **시간 복잡도** 최악 O(N²). 그래서 conf thresholding으로 후보부터 줄임.
- **클래스별 NMS**: 이 구현은 클래스 구분 없음(class-agnostic). 실제 YOLO는 클래스마다 따로 NMS → 사람 박스가 그 위의 자전거 박스를 지우지 않음.
- NMS가 남긴 중복 박스 = **교집합은 있지만 합집합이 커서 IoU가 임계값보다 낮은 경우** (크기가 크게 다른 두 박스 등).

---

## 제출 체크리스트

- [ ] `03_DL-and-GPU.ipynb` — G-4) 잔상 효과 CPU/GPU 함수, 업로드/다운로드 셀 완성 + 결과 이미지 출력
- [ ] `03_DL-and-GPU.ipynb` — G-5) `convolution2d_cp`, CPU 처리, GPU 처리, 결과 비교 셀 완성
- [ ] `03_DL-and-GPU.ipynb` — I-3) `main_gpu()` 완성 + CPU vs GPU 비교 출력
- [ ] `03_DL-and-GPU.ipynb` — J-12) CIFAR-10 학습 루프 (버그 4개 수정 반영)
- [ ] `src/models/MNIST/MNIST_CNN.pth` 생성 확인
- [ ] `src/models/CIFAR10/CIFAR10_CNN.pth` 생성 + 로드 후 정확도 출력 확인
- [ ] `04_DL-Object-Detection.ipynb` — B) `convert_bbox_coord`, `calculate_iou` + 3장 IoU 출력
- [ ] `04_DL-Object-Detection.ipynb` — C) `prune_low_conf`, `apply_nms` + 적용 셀 연결 수정 + cv2 결과 비교
- [ ] 본 README의 결과 표 `__` 빈칸 채우기

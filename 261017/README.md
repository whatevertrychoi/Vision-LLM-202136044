# Vision-LLM-202136044

컴퓨터 비전 실습 코드 저장소입니다. 이 문서는 `02_Computer-vision.py`의 **농구공 영역 검출** 단계를 설명합니다.

## 개요

`02_Computer-vision.py`는 색상 기반으로 만든 마스크(`mask_enhanced`)에서 노이즈를 제거하고, 농구공 영역을 하나의 외곽선으로 추출해 원본 이미지 위에 시각화합니다.

## 처리 흐름

| 단계 | 내용 | 사용 함수 |
|------|------|-----------|
| 1 | 타원형 구조 요소(5×5) 생성 | `cv2.getStructuringElement` |
| 2 | Opening (2회) — 작은 노이즈 제거 | `cv2.morphologyEx(..., MORPH_OPEN)` |
| 3 | Closing (2회) — 내부 구멍 메우기 | `cv2.morphologyEx(..., MORPH_CLOSE)` |
| 4 | 최외곽 외곽선 검출 | `cv2.findContours` (`RETR_EXTERNAL`) |
| 5 | 면적이 가장 큰 외곽선을 농구공으로 선택 | `max(contours, key=cv2.contourArea)` |
| 6 | 마스크를 초록색으로 반투명 오버레이 + 외곽선 표시 | `cv2.addWeighted`, `cv2.drawContours` |

## 입력 / 출력

- **입력**
  - `mask_enhanced`: 이전 단계에서 생성한 binary mask
  - `basketball_enhanced_rgb`: 시각화에 사용할 RGB 이미지
- **출력**
  - `ball_mask_final`: Morphological 연산이 적용된 최종 binary mask
  - `ball_img`: 농구공 영역과 외곽선이 표시된 이미지

## 시각화 결과

`matplotlib`으로 세 장을 나란히 출력합니다.

1. Enhanced Mask — 입력 마스크
2. Mask (open x2, close x2) — Morphological 연산 후 마스크
3. Largest Contour — 가장 큰 외곽선을 표시한 결과

## 환경

```bash
pip install opencv-python numpy matplotlib
```

## 실행

```bash
python 02_Computer-vision.py
```

## 참고

- Opening은 침식 후 팽창이라 작은 점 노이즈를 지우고, Closing은 팽창 후 침식이라 물체 내부의 빈 구멍을 메웁니다.
- 가장 큰 외곽선만 사용하므로, 배경에 농구공보다 큰 비슷한 색 영역이 있으면 잘못 선택될 수 있습니다.

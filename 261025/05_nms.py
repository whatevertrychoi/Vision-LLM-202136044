# TODO: Confidence Thresholding 함수 정의
# TODO: Confidence Score가 특정 임계값보다 낮으면 해당 BBox 제거

def prune_low_conf(boxes, scores, conf_threshold=0.5):
    """
    Confidence Thresholding 적용

    Parameters
    ----------
    boxes : list
        Bounding Box 목록
        각 박스는 (x1, y1, x2, y2) 형식

    scores : list
        각 Bounding Box의 Confidence Score

    conf_threshold : float
        Bounding Box를 유지하기 위한 최소 Confidence Score

    Returns
    -------
    pruned_boxes : list
        Confidence Score가 임계값 이상인 Bounding Box 목록

    pruned_scores : list
        Confidence Score가 임계값 이상인 Bounding Box의 Confidence Score 목록
    """
    pruned_boxes, pruned_scores = [], []
    for box, score in zip(boxes, scores):
        if score >= conf_threshold:
            pruned_boxes.append(box)
            pruned_scores.append(score)

    return pruned_boxes, pruned_scores

# TODO: NMS 함수 정의
# TODO: np.sort() 함수를 사용하면 1차원 배열을 정렬
# TODO: np.argsort() 함수를 사용하면 1차원 배열의 정렬된 인덱스를 반환

def apply_nms(boxes, scores, iou_threshold=0.5):
    """
    NMS 적용

    Parameters
    ----------
    boxes : list
        Bounding Box 목록
        각 박스는 (x1, y1, x2, y2) 형식

    scores : list
        각 Bounding Box의 Confidence Score

    iou_threshold : float
        중복 Bounding Box를 제거할 IoU 기준값

    Returns
    -------
    nms_boxes : list
        NMS 적용 후 유지된 Bounding Box 목록

    nms_scores : list
        NMS 적용 후 유지된 Bounding Box의 Confidence Score 목록
    """
    # 입력 박스가 없으면 바로 빈 결과 반환
    if len(boxes) == 0:
        return [], []

    # ── 1. score 내림차순으로 인덱스 정렬 ──
    # np.argsort: 오름차순 정렬된 "인덱스" 반환
    # [::-1]: 뒤집어서 내림차순 (score 높은 박스가 앞)
    # .tolist(): pop / 리스트 컴프리헨션 쓰기 위해 파이썬 리스트로 변환
    # 예) scores = [0.7, 0.9, 0.6] → argsort = [2, 0, 1] → 뒤집기 = [1, 0, 2]
    order = np.argsort(scores)[::-1].tolist()

    # 최종적으로 살아남을 박스들의 인덱스
    keep = []

    # ── 2. 남은 후보가 없을 때까지 반복 ──
    while order:
        # 남은 후보 중 score 최고인 박스를 꺼내서 확정
        best = order.pop(0)
        keep.append(best)

        # ── 3. best와 많이 겹치는 박스 제거 ──
        # IoU > threshold → 같은 객체를 중복 검출한 것으로 보고 버림
        # IoU <= threshold → 다른 객체일 수 있으므로 다음 라운드 후보로 유지
        order = [
            i for i in order
            if calculate_iou(boxes[best], boxes[i]) <= iou_threshold
        ]

    # ── 4. 살아남은 인덱스로 박스·점수 추출 ──
    # keep은 이미 score 내림차순 순서
    nms_boxes = [boxes[i] for i in keep]
    nms_scores = [scores[i] for i in keep]

    return nms_boxes, nms_scores
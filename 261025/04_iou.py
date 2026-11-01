# TODO: 1) BBox 좌표를 (x,y,w,h)에서 (x1,y1,x2,y2)로 변환하는 함수 정의
# TODO: 2) IoU 계산 함수 정의

def convert_bbox_coord(bbox):
    """
    (x, y, w, h) → (x1, y1, x2, y2)
    - (x, y): 박스의 좌상단 좌표 (COCO 형식)
    - (x1, y1): 좌상단, (x2, y2): 우하단
    이미지 좌표계는 y축이 아래로 증가하므로 y + h가 아래쪽 경계
    """
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def calculate_iou(bbox_a, bbox_b):
    """
    IoU (Intersection over Union) = 교집합 면적 / 합집합 면적
    - 두 박스가 얼마나 겹치는지 0~1로 표현 (1 = 완전히 일치, 0 = 안 겹침)
    - 입력은 (x1, y1, x2, y2) 형식이어야 함
    """
    xa_1, ya_1, xa_2, ya_2 = bbox_a
    xb_1, yb_1, xb_2, yb_2 = bbox_b

  
    # 좌상단: 두 박스 좌상단 중 더 안쪽(오른쪽·아래쪽) → max
    intersection_x1 = max(xa_1, xb_1)
    intersection_y1 = max(ya_1, yb_1)
    # 우하단: 두 박스 우하단 중 더 안쪽(왼쪽·위쪽) → min
    intersection_x2 = min(xa_2, xb_2)
    intersection_y2 = min(ya_2, yb_2)

    
    # 안 겹치면 x2 < x1 (또는 y2 < y1) → 음수가 나오므로 0으로 클리핑
    # (음수 × 음수 = 양수가 되는 오류 방지)
    w = max(0, intersection_x2 - intersection_x1)
    h = max(0, intersection_y2 - intersection_y1)
    intersection_area = w * h

    # ── 3. 각 박스 면적 ──
    area_a = (xa_2 - xa_1) * (ya_2 - ya_1)
    area_b = (xb_2 - xb_1) * (yb_2 - yb_1)

    
    # 교집합이 두 번 더해지므로 한 번 빼줌
    union_area = area_a + area_b - intersection_area

    # 두 박스 모두 면적 0(w=0 또는 h=0)인 퇴화 박스일 때 발생
    if union_area <= 0:
        return 0.0

    # TODO: ZeroDivisionError 주의 → 위 조건문으로 처리됨
    return intersection_area / union_area
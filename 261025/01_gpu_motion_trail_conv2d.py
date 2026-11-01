import cupy as cp

# TODO: CuPy를 사용하여 잔상 효과 함수 정의

def motion_trail_cp(image, weights):
    """
    CuPy를 이용해 이미지 잔상 효과를 생성한다.

    이동 거리는 Python에서 순차적으로 반복하지만,
    각 이동 거리의 이미지 전체 계산은 GPU에서 병렬 처리한다.
    """

    # TODO: 필요한 변수 정의 및 출력 배열 생성
    height, width = image.shape
    trail_length = min(TRAIL_LENGTH, weights.shape[0])
    output = cp.zeros((height, width), dtype=cp.float32)

    for shift in range(TRAIL_LENGTH):
        if shift >= trail_length:
            break

        # TODO: 배열 slicing
        input_region = image[:, :width - shift]   # 오른쪽 shift칸 제외한 입력
        output_region = output[:, shift:]         # 왼쪽 shift칸 제외한 출력 (view)

        # TODO: GPU가 병렬 연산할 output_region 계산 코드
        output_region += weights[shift] * input_region

    return output


# TODO: CuPy를 사용하여 2차원 convolution 함수 정의

def convolution2d_cp(image, kernel, bias=0.0, stride=1, padding=0):
    """
    CuPy를 이용한 2차원 convolution.

    커널의 위치는 Python 반복문으로 순차 처리하고,
    각 위치에 해당하는 출력 배열 전체 연산은 GPU에서 처리한다.
    """

    kernel_height, kernel_width = kernel.shape

    # 이미지 가장자리에 zero padding 적용 (GPU)
    padded_image = cp.pad(
        image,
        ((padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0
    )

    # 출력 크기 = floor((H + 2P - K) / S) + 1
    output_height = (padded_image.shape[0] - kernel_height) // stride + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1

    # 출력 배열 준비 (GPU)
    output = cp.zeros((output_height, output_width), dtype=cp.float32)

    # 커널 원소 하나씩 선택 (3x3이면 9번 반복)
    for kernel_y in range(kernel_height):
        for kernel_x in range(kernel_width):
            # 모든 출력 위치에서 현재 커널 원소와 대응되는 입력 픽셀을 한 번에 선택
            # 시작: kernel_y, 간격: stride, 개수: output_height개
            input_region = padded_image[
                kernel_y : kernel_y + stride * (output_height - 1) + 1 : stride,
                kernel_x : kernel_x + stride * (output_width - 1) + 1 : stride
            ]

            # 배열 전체에 현재 커널 가중치를 곱해 출력 배열 전체에 누적 (GPU 병렬)
            output += kernel[kernel_y, kernel_x] * input_region

    # 편향 적용
    output += bias

    return output
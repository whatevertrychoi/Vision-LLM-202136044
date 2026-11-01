import time
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# TODO: CUDA GPU 장치에서 CIFAR-10 데이터셋을 학습과 평가
# TODO: 각 Epoch 및 전체 학습 시간을 출력

EPOCHS = 5  # 전체 학습 데이터를 5번 반복해서 학습

# 학습 기록 저장 공간 준비 (epoch별 값 → 학습 곡선 그래프용)
train_loss_history = []
train_accuracy_history = []
test_loss_history = []
test_accuracy_history = []

# 전체 학습 시간 측정 시작
# 이전 셀에서 남은 GPU 작업(모델 .to(device) 등)이 끝난 뒤 시각을 찍기 위해 동기화
torch.cuda.synchronize()
training_start_time = time.perf_counter()

# TODO: 5번의 Epoch 동안 학습 및 평가 진행
for epoch in range(EPOCHS):

    # epoch 시작 시각 (GPU 대기열 비운 뒤 측정)
    torch.cuda.synchronize()
    epoch_start_time = time.perf_counter()

    # 학습: 학습셋(50,000장)으로 forward → loss → backward → 가중치 업데이트
    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,   
        criterion,
        optimizer,
        device,
    )

    # 평가: 테스트셋(10,000장)으로 성능만 측정 (가중치 업데이트 X)
    test_loss, test_accuracy = evaluate(  
        model,
        test_loader,
        criterion,
        device,
    )

    # epoch 결과 기록
    train_loss_history.append(train_loss)
    train_accuracy_history.append(train_accuracy)
    test_loss_history.append(test_loss)
    test_accuracy_history.append(test_accuracy)

    # GPU 비동기 실행 → 모든 커널이 끝난 뒤 epoch 시간 계산
    torch.cuda.synchronize()
    epoch_time = time.perf_counter() - epoch_start_time

    # epoch 결과 출력 (accuracy는 0~1 비율 → ×100 해서 %)
    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train accuracy: "
        f"{train_accuracy * 100:.2f}% | "
        f"Test loss: {test_loss:.4f} | "
        f"Test accuracy: "
        f"{test_accuracy * 100:.2f}% | "
        f"Time: {epoch_time:.2f}s"
    )

# TODO: 전체 학습 시간 계산 및 출력
torch.cuda.synchronize()
total_training_time = time.perf_counter() - training_start_time
print(f"Total GPU training time: {total_training_time:.2f}s")
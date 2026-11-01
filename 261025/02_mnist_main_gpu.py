import time
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# TODO: GPU 기반 MNIST 데이터셋 학습 코드 작성

def main_gpu():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # 전처리
    transform = transforms.ToTensor()

    #데이터셋 로드
    #학습
    train_dataset = datasets.MNIST(
        root=DATA_ROOT,
        train=True,
        transform=transform,
        download=True,
    )
    #테스트
    test_dataset = datasets.MNIST(
        root="src/datasets",
        train=False,
        transform=transform,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    #device 설정 및 모델 배치
    device = torch.device("cuda")

    model = SimpleCNN()
    model = model.to(device)

    # 손실 함수 & 옵티마이저
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    train_loss_history = []
    train_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []

    torch.cuda.synchronize()
    training_start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        epoch_start_time = time.perf_counter()
        
        # 1 epoch 학습 : forward → loss → backward → optimizer.step 반복
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        # 테스트셋 평가: 가중치 업데이트 없이 성능만 측정
        # (내부에서 model.eval() + torch.no_grad() 사용하는 것이 정석)
        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        # epoch 별 결과 저장 
        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

        # GPU 커널은 비동기 → 모든 연산이 실제로 끝난 뒤에 시간 측정
        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start_time

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
    torch.cuda.synchronize()
    total_training_time = time.perf_counter() - training_start_time
    print(f"Total GPU training time: {total_training_time:.2f}s")

    return total_training_time
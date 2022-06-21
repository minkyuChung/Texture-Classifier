import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# 이미지 패치에서 특징 추출
train_dir = './texture_data/train'
test_dir = './texture_data/test'
classes = ['brick','grass','ground','water','wood']

X_train = []
Y_train = []

PATCH_SIZE = 32
np.random.seed(1234)
for idx, texture_name in enumerate(classes):
    image_dir = os.path.join(train_dir, texture_name)                       # class image가 있는 경로
    for image_name in os.listdir(image_dir):
        image = cv2.imread(os.path.join(image_dir, image_name))             # 이미지 불러오기
        image_s = cv2.resize(image, (100,100), interpolation=cv2.INTER_LINEAR)  # 이미지를 100x100으로 축소
        
        for _ in range(10):
            h = np.random.randint(100-PATCH_SIZE)
            w = np.random.randint(100-PATCH_SIZE)
            
            image_p = image_s[h:h+PATCH_SIZE, w:w+PATCH_SIZE]               # 이미지 패치 자르기
            
            X_train.append(image_p)                                         # laws texture 특징 추가(9차원)
            Y_train.append(idx)                                             # 라벨 추가
            
X_train = np.array(X_train)/128-1                                           # list를 numpy array로 변경
X_train = np.swapaxes(X_train, 1, 3)                                        # (N, Cin, H, W)
Y_train = np.array(Y_train)
print('train data: ', X_train.shape)                                        # (300, 3, 32, 32)
print('train label: ', Y_train.shape)                                       # (300, 3)

X_test = []
Y_test = []

for idx, texture_name in enumerate(classes):
    image_dir = os.path.join(test_dir, texture_name)                        # class image가 있는 경로
    for image_name in os.listdir(image_dir):                                # 경로에 있는 모든 이미지에 대해
        image = cv2.imread(os.path.join(image_dir, image_name))             # 흑백으로 불러오기
        X_test.append(image)                                                # laws texture 특징 추가 (9차원)
        Y_test.append(idx)                                                  # 라벨 추가
        
X_test = np.array(X_test)/128 - 1                                           # list를 numpy array로 변경
X_test = np.swapaxes(X_test, 1, 3)
Y_test = np.array(Y_test)
print('test data: ', X_test.shape)                                          # (150, 3, 32, 32)
print('test label: ', Y_test.shape)                                         # (150, 3)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchsummary import summary

class Dataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.images[idx]
        label = self.labels[idx]
        sample = (image, label)
        
        return sample
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=10, out_channels=3, kernel_size=3)
        self.pool2 = nn.AdaptiveMaxPool2d(output_size=3)
        self.fc1 = nn. Linear(27, 5)
        self.relu = nn.ReLU6()
        
    def forward(self, x):
        
        out = self.conv1(x)                 # 10x30x30
        out = self.relu(out)                # 10x30x30
        out = self.conv2(out)               # 10x28x28
        out = self.relu(out)                # 10x28x28
        out = self.pool1(out)               # 10x14x14
        out = self.conv3(out)               # 10x12x12
        out = self.relu(out)                # 10x12x12
        out = self.conv4(out)               # 3x10x10
        out = self.relu(out)                # 3x10x10
        out = self.pool2(out)               # 3x3x3
        out = torch.flatten(out, 1)         # 27
        out = self.fc1(out)                 # 3
        
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       # GPU: 'cuda', CPU: 'cpu'

batch_size = 10
learning_rate = 0.001
n_epoch = 100

Train_data = Dataset(images=X_train, labels=Y_train)
Test_data = Dataset(images=X_test, labels=Y_test)

Trainloader = DataLoader(Train_data, batch_size=batch_size, shuffle=True)   # 학습 데이터 로더 정의
Testloader = DataLoader(Test_data, batch_size=batch_size)

net = CNN()
net.to(device)                                                              # 모델을 device로 보내기
summary(net,(3, 32, 32), device='cuda' if torch.cuda.is_available() else 'cpu')   # 모델 layer 출력

optimizer = optim.Adam(net.parameters(), lr=learning_rate)                   # 옵티마이저 정의
criterion = nn.CrossEntropyLoss()                                           # loss 계산식 정의

train_losses = []           # 학습 loss를 저장할 list정의
train_accs = []             # 학습 accuracy를 저장할 list 정의
test_losses = []            # 테스트 loss
test_accs = []

# 학습
for epoch in range(n_epoch):
    train_loss = 0.0
    evaluation = []
    for i, data in enumerate(Trainloader, 0):
        features, labels = data         # 데이터를 특징과 라벨로 나누기
        labels = labels.long().to(device)
        features = features.to(device)                                      # 특징을 device로 보내기
        optimizer.zero_grad()                                               # optimizer의 gradient를 0으로 초기화
        
        outputs = net(features.to(torch.float))                             # 특징을 float형으로 변환후 모델에 입력
        
        _, predicted = torch.max(outputs.cpu().data, 1)                     # 출력의 제일 큰 값의 index 반환
        
        evaluation.append((predicted==labels.cpu()).tolist())               # 정답과 비교하여 True False 값을 저장
        loss = criterion(outputs, labels)                                   # 출력과 라벨을 비교하여 loss 계산
        
        loss.backward()                                                     # 역전파, 기울기 계산
        optimizer.step()                                                    # 가중치 값 업데이트, 학습 한번 진행
        
        train_loss += loss.item()                                           # loss를 train_loss에 누적
    train_loss = train_loss/(i+1)                                           # 평균 train_loss 구하기
    evaluation = [item for sublist in evaluation for item in sublist]       # [True, false] 값을 list로 저장, 300차원
    train_acc = sum(evaluation)/len(evaluation)                             # True인 비율 계산
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # 테스트
    if (epoch+1) % 1 == 0:
        test_loss = 0.0
        evaluation = []
        for i, data in enumerate(Testloader, 0):
            features, labels = data
            labels = labels.long().to(device)
            
            features = features.to(device)
            
            outputs = net(features.to(torch.float))                         # 특징을 float형으로 변환 후 모델에 입력
            _, predicted = torch.max(outputs.cpu().data, 1)                 # 출력의 제일 큰 값의 index 반환
            evaluation.append((predicted==labels.cpu()).tolist())           # 정답과 비교하여 True False값을 저장
            loss = criterion(outputs, labels)                               # 출력과 라벨을 비교하여 loss 계산
            test_loss += loss.item()                                        # test를 test_loss에 누적
        test_loss = test_loss/(i+1)                                         # 평균 구하기
        evaluation = [item for sublist in evaluation for item in sublist]   # [True, False] 값을 list로 저장, 150차원
        test_acc = sum(evaluation)/len(evaluation)                          # True인 비율 계산
        
        test_losses.append(test_loss)                                       # 해당 epoch의 test loss 기록
        test_accs.append(test_acc)                                          # 해당 epoch의 test acc 기록
        
        print('[%d, %3d]\tloss: %.4f\tAccuracy : %.4f\t\tval-loss: %.4f\tval-Accuracy : %.4f' %
              (epoch+1, n_epoch, train_loss, train_acc, test_loss, test_acc))

# 학습/테스트 loss/정확도 시각화
plt.plot(range(len(train_losses)), train_losses, label='train loss')
plt.plot(range(len(test_losses)), test_losses, label='test loss')
plt.legend()
plt.show()
plt.plot(range(len(train_accs)), train_accs, label='train acc')
plt.plot(range(len(test_accs)), test_accs, label='test acc')
plt.legend()
plt.show()





















from skimage.feature import greycomatrix, greycoprops
import matplotlib.pyplot as plt
from scipy import signal as sg
import numpy as np
import cv2
import os

# laws texture 계산 함수
def laws_texture(gray_image):
    (rows,cols) = gray_image.shape[:2]
    smooth_kernel = (1/25)*np.ones((5,5))                               # smoothing filter 만들기
    gray_smooth = sg.convolve(gray_image, smooth_kernel, "same")        # 흑백이미지 smoothing하기
    gray_processed = np.abs(gray_image - gray_smooth)                   # 원본이미지에서 smoothing된 이미지 빼기
    
    filter_vectors = np.array([[ 1, 4, 6, 4, 1],            #L5
                               [-1,-2, 0, 2, 1],            #E5
                               [-1, 0, 2, 0, 1],            #S5
                               [ 1,-4, 6,-4, 1]])           #R5
    filters = []                                            #16(4x4)개 filter를 저장할 filters
    for i in range(4):
        for j in range(4):
            filters.append(np.matmul(filter_vectors[i][:].reshape(5,1), # 매트릭스 곱하기 연산을 통해 filter 값 계산
                                     filter_vectors[j][:].reshape(1,5)))
    
    conv_maps = np.zeros((rows, cols, 16))                              # 계산된 convolution 결과 저장할 conv_maps
    for i in range(len(filters)):
        conv_maps[:,:,i] = sg.convolve(gray_processed,                  # 전처리된 이미지에 16개 필터 적용
                                       filters[i],'same')
        
    # 9+1개 중요한 texture map 계산
    texture_maps = list()
    texture_maps.append((conv_maps[:,:,1]+conv_maps[:,:,4])//2)         # L5E5 / E5L5
    texture_maps.append((conv_maps[:,:,2]+conv_maps[:,:,8])//2)         # L5S5 / S5L5
    texture_maps.append((conv_maps[:,:,3]+conv_maps[:,:,12])//2)        # L5R5 / R5L5
    texture_maps.append((conv_maps[:,:,7]+conv_maps[:,:,13])//2)        # E5R5 / R5E5
    texture_maps.append((conv_maps[:,:,6]+conv_maps[:,:,9])//2)         # E5S5 / S5E5
    texture_maps.append((conv_maps[:,:,11]+conv_maps[:,:,14])//2)       # S5R5 / R5S5
    texture_maps.append(conv_maps[:,:,10])                              # S5S5
    texture_maps.append(conv_maps[:,:,5])                               # E5E5
    texture_maps.append(conv_maps[:,:,15])                              # R5R5
    texture_maps.append(conv_maps[:,:,0])                               # L5L5
    
    # law's texture energy 계산
    TEM = list()
    for i in range(9):
        TEM.append(np.abs(texture_maps[i]).sum()/                       # TEM 계산 및 L5L5 값으로 정규화
                   np.abs(texture_maps[9]).sum())
    return TEM

# 이미지 패치에서 특징 추출
train_dir = './texture_data/train'
test_dir = './texture_data/test'
classes = ['brick','grass','ground','water','wood']

X_train = []
Y_train = []

PATCH_SIZE = 30
np.random.seed(1234)
for idx, texture_name in enumerate(classes):
    image_dir = os.path.join(train_dir, texture_name)                       # class image가 있는 경로
    for image_name in os.listdir(image_dir):                                # 경로에 있는 모든 이미지에 대해
        image = cv2.imread(os.path.join(image_dir, image_name))             # 이미지 불러오기
        image_s = cv2.resize(image,(100,100),interpolation=cv2.INTER_LINEAR)    # 이미지 100X100으로 축소
        
        for _ in range(10):
            h = np.random.randint(100-PATCH_SIZE)                           # 랜덤하게 자를 위치 선정
            w = np.random.randint(100-PATCH_SIZE)
            
            image_p = image_s[h:h+PATCH_SIZE, w:w+PATCH_SIZE]               # 이미지 패치 자르기
            image_p_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)          # 이미지 흑백으로 변화
            #image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV))            # 이미지 HSV로 변환
            glcm = greycomatrix(image_p_gray, distances=[1], angles=[0], levels = 256,  # GLCM co-occurence 계산
                                symmetric=False, normed=True)
            X_train.append([greycoprops(glcm, 'dissimilarity')[0,0],
                            greycoprops(glcm, 'correlation')[0,0],
                            greycoprops(glcm, 'homogeneity')[0,0]]
                           # GLCM dissimilarity, correlation 특징 추가(5차원)
                           + laws_texture(image_p_gray))                    # laws texture 특징 추가(9차원)
            Y_train.append(idx)                                             # 라벨 추가
            
X_train = np.array(X_train)                                                 # list를 numpy array로 변경
Y_train = np.array(Y_train)
print('train data: ', X_train.shape)                                        # (300,11)
print('train label: ', Y_train.shape)                                       # (300)

X_test = []
Y_test = []

for idx, texture_name in enumerate(classes):
    image_dir = os.path.join(test_dir, texture_name)                        # class image가 있는 경로
    for image_name in os.listdir(image_dir):                                #경로에 있는 모든 이미지에 대해
        image = cv2.imread(os.path.join(image_dir,image_name))              # 흑백으로 불러오기
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        glcm = greycomatrix(image_gray, distances=[1], angles=[0], levels=256,  # glcm co-occurence 계산
                            symmetric=False, normed=True)
        X_test.append([greycoprops(glcm,'dissimilarity')[0,0],
                       greycoprops(glcm,'correlation')[0,0],
                       greycoprops(glcm, 'homogeneity')[0,0]]
                      + laws_texture(image_gray))                           # laws texture 특징 추가 (9차원)
        Y_test.append(idx)
        
X_test = np.array(X_test)
Y_test = np.array(Y_test)
print('test data: ', X_test.shape)                                          # (150,11)
print('test label: ', Y_test.shape)                                         # (150)
    
# 신경망에 필요한 모듈
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchsummary import summary

# 데이터셋 클래스
class textureDataset(Dataset):                                              # 데이터셋 클래스
    def __init__(self,features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):                                             # idx 번째 샘플 반환
        if torch.is_tensor(idx):                                            # idx가 pytorch tensor면
            idx = idx.tolist()                                              # idx를 list로 변환
        feature = self.features[idx]
        label = self.labels[idx]
        sample = (feature, label)                                           # idx번째 특징과 라벨을 샘플로 묶어 반환            
        
        return sample
    
# 신경망 모델 클래스
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()                                         # 기반 클래스 nn.Module을 초기화
        self.fc1 = nn.Linear(input_dim, hidden_dim)                         # input_dim X hidden_dim
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)                        # hidden_dim X hidden_dim
        self.fc3 = nn.Linear(hidden_dim, output_dim)                        # hidden_dim x output_dim
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        
        return out
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       # GPU: 'cuda', CPU: 'cpu'

batch_size = 10
learning_rate = 0.01
n_epoch = 500

Train_data = textureDataset(features=X_train, labels=Y_train)
Test_data = textureDataset(features=X_test, labels=Y_test)

Trainloader = DataLoader(Train_data, batch_size=batch_size, shuffle=True)   # 학습 데이터 로더 정의
Testloader = DataLoader(Test_data, batch_size=batch_size)

net = MLP(12,8,5)                           # input 차원에 영향 주는 코드
net.to(device)                                                              # 모델을 device로 보내기
summary(net,(12,), device='cuda' if torch.cuda.is_available() else 'cpu')   # 모델 layer 출력

optimizer = optim.SGD(net.parameters(), lr=learning_rate)                   # 옵티마이저 정의
criterion = nn.CrossEntropyLoss()                                           # loss 계산식 정의

train_losses = []           # 학습 loss를 저장할 list정의
train_accs = []             # 학습 accuracy를 저장할 list 정의
test_losses = []            # 테스트 loss
test_accs = []
    
# 학습
for epoch in range(n_epoch):
    train_loss = 0.0
    evaluation = []
    net.train()                         # 학습모드로 전환
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
        net.eval()                                                          # 모델을 평가 모드로 전환
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

    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
import numpy as np
import cv2 
from matplotlib import pyplot as plt
from scipy import signal as sg

# 이미지 읽기
image = cv2.imread('pebbles.jpg')                           # 이미지를 컬러로 불러오기
image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)             # BGR -> RGB
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)              # BGR -> 흑백
(rows, cols) = gray.shape[:2]                               # 이미지 길이와 너비

# 이미지 전처리
smooth_kernel = (1/25)*np.ones((5,5))                       # smoothing filter 만들기
gray_smooth = sg.convolve(gray, smooth_kernel, "same")      # 흑백이미지 smoothing
                                                            # 출력이미지 사이즈 = 입력이미지 사이즈
gray_processed = np.abs(gray - gray_smooth)                 # 원본이미지에서 smoothing된 이미지 빼기

# 전처리된 이미지 시각화
plt.figure('Image pre-processing')                          # 그림판 만들기
plt.subplot(221)                                            # 그림판을 2행2열로 나누고 1번 영역에 그림 그리기
plt.title('Image')
plt.imshow(image2)
plt.subplot(222)
plt.title('Gray')
plt.imshow(gray,'gray')
plt.subplot(223)
plt.title('Smoothing')
plt.imshow(gray_smooth, 'gray')
plt.subplot(224)
plt.title('Subtract smoothing')
plt.imshow(gray_processed, 'gray')

# law's texture filter
filter_vectors = np.array([[ 1, 4, 6, 4, 1],                    # L5
                           [-1,-2, 0, 2, 1],                    # E5
                           [-1, 0, 2, 0, 1],                    # S5
                           [ 1,-4, 6,-4, 1]])                   # R5

                                                                # 0:L5L5, 1:L5E5, 2:L5S5, 3:L5R5,
                                                                # 4:E5L5, 5:E5E5, 6:E5S5, 7:E5R5,
                                                                # 8:S5L5, 9:S5E5, 10:S5S5, 11:S5R5,
                                                                # 12:R5L5, 13:R5E5, 14:R5S5, 15:R5R5
filters = list()                                                # 16(4x4)개 filter를 저장할 filters
for i in range(4):
    for j in range(4):
        filters.append(np.matmul(filter_vectors[i][:].reshape(5,1),  # 매트릭스 곱하기 연산을 통해 filter값 계산
                                 filter_vectors[j][:].reshape(1,5)))

# convolution 연산 및 convmap 조합
conv_maps = np.zeros((rows, cols, 16))                          # 계산된 convolution 결과를 저장할 conv_maps
for i in range(len(filters)):
    conv_maps[:,:,1] = sg.convolve(gray_processed,
                                   filters[i],'same')           # 전처리된 이미지에 16개 필터 적용

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
print(TEM)                                                          # 9차원의 TEM feature 추출

# 결과 시각화
def norm(ar):
    """ Convolution된 이미지를 0~255로 정규화"""
    return 255.*np.absolute(ar)/np.max(ar)

plt.figure('Texture Maps')
plt.subplot(331)
plt.title('L5E5 / E5L5')
plt.imshow(norm(texture_maps[0]),'gray')
plt.subplot(332)
plt.title('L5S5 / S5L5')
plt.imshow(norm(texture_maps[1]),'gray')
plt.subplot(333)
plt.title('L5R5 / R5L5')
plt.imshow(norm(texture_maps[2]),'gray')
plt.subplot(334)
plt.title('E5R5 / R5E5')
plt.imshow(norm(texture_maps[3]),'gray')
plt.subplot(335)
plt.title('E5S5 / S5E5')
plt.imshow(norm(texture_maps[4]),'gray')
plt.subplot(336)
plt.title('S5R5 / R5S5')
plt.imshow(norm(texture_maps[5]),'gray')
plt.subplot(337)
plt.title('S5S5')
plt.imshow(norm(texture_maps[6]),'gray')
plt.subplot(338)
plt.title('E5E5')
plt.imshow(norm(texture_maps[7]),'gray')
plt.subplot(339)
plt.title('R5R5')
plt.imshow(norm(texture_maps[8]),'gray')
plt.show()
































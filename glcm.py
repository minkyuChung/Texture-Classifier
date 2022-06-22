import matplotlib.pyplot as plt
import cv2
from skimage.feature import greycomatrix, greycoprops
import numpy as np

# 이미지 읽기
image = cv2.imread('camera.png',cv2.IMREAD_GRAYSCALE)           # 이미지를 흑백으로 불러오기

# 이미지에서 풀과 하늘 영역 잘라내기
PATCH_SIZE = 21                                                 # 이미지에서 잘라낼 영역 너비와 길이 (너비 = 길이 = PATCH_SIZE)

grass_locations = [(370, 454), (372,22), (444,244), (455,455)]  # 이미지에서 풀 부분 위치 선정, 이미지 좌측 상단이 원점 기준(y축, x축)
grass_patches = list()
for loc in grass_locations:
    grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                               loc[1]:loc[1] + PATCH_SIZE])
    
sky_locations = [(38,34), (139,28), (37,437), (145, 379)]       # 이미지에서 하늘 부분 위치 선정, 이미지 좌측 상단이 원점 기준(y축, x축)
sky_patches = list()
for loc in sky_locations:
    sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                             loc[1]:loc[1] + PATCH_SIZE])

# 잘라낸 풀과 하늘 영역에서 GLCM dissimilarity와 correlation 계산하기
xs = list()                                                     # dissimilarity(entropy)값 저장할 list
ys = list()                                                     # correlation값 저장할 list
for patch in (grass_patches + sky_patches):
    glcm = greycomatrix(patch, distances=[1], angles=[0], levels=256,   # GLCM co-occurence 계산
                        symmetric=False, normed=True)
    xs.append(greycoprops(glcm, 'dissimilarity')[0,0])          # GLCM dissimilarity(entropy)값 계산하여 xs에 저장
    ys.append(greycoprops(glcm, 'correlation')[0,0])            # GLCM correlation 계산하여 ys에 저장
                                                                # 그외에도 다른 feature 계산 가능
                                                                # {'contrast', 'dissimilarity', 'homogeneity',
                                                                # 'energy', 'correlation', 'ASM'}
# 결과 시각화
fig = plt.figure(figsize=(8,8))                                 # 그림판 만들기

ax = fig.add_subplot(3,2,1)                                     # 그림판을 3행2열로 나누고 1번 영역에 그림 그리기
ax.imshow(image, cmap=plt.cm.pray, vmin=0, vmax=255)            # 흑백이미지 추가
for(y,x) in grass_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')       # 중심에 초록색 네모(green square)
for(y,x) in sky_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')       # 중심에 하늘색 네모
ax.set_xlabel('Original Image')                                 # x축 이름을 Original Image로

ax = fig.add_subplot(3,2,2)                                     # 그림판을 3행 2열로 나누고 2번 영역에 그림그리기
ax.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go', # dissimilarity값을 x축, correlation값을 y축인 지점에 초록색 점 추가
        label='Grass')
ax.plot(xs[len(sky_patches):], ys[len(sky_patches):], 'bo',     # dissimilarity값을 x축, correlation값을 y축인 지점에 하늘색 점 추가
        label='Sky')
ax.set_xlabel('GLCM Dissimilarity')                             # x축 이름을 GLCM Dissimilarity으로 지정
ax.set_ylabel('GLCM Correlation')                               # y축 이름을 GLCM Correlation으로 지정
ax.legend()                                                     # 라벨 이름 추가


for i, patch in enumerate(grass_patches):                       # 각 풀 영역마다
    ax = fig.add_subplot(3, len(grass_patches),                 # 그림판을 3행4열로 나누고
                         len(grass_patches)*1 + i + 1)          # 5,6,7,8번 영역에 그림그리기
    ax.imshow(patch, cmap=plt.cm.gray,                          # 풀 영역 추가하기
              vmin=0, vmax=255)                         
    ax.set_xlabel('Grass %d' %(i + 1))                          # 각 풀 영역 이름 지정

for i, patch in enumerate(sky_patches):                         # 각 하늘 영역마다
    ax = fig.add_subplot(3, len(sky_patches),                   # 그림판을 3행4열로 나누고
                         len(sky_patches)*1 + i + 1)            # 9,10,11,12번 영역에 그림그리기
    ax.imshow(patch, cmap=plt.cm.gray,                          # 하늘 영역 추가하기
              vmin=0, vmax=255)                         
    ax.set_xlabel('Sky %d' %(i + 1))                            # 각 하늘 영역 이름 지정

plt.tight_layout()                                              # 여백공간 설정
plt.show()                                                      # 그림 시각화


















import matplotlib.pyplot as plt
import cv2
from skimage.feature import greycomatrix, greycoprops
import numpy as np
import random

# brick에 대한 train 수행

brick_xs = []
brick_ys = []
brick_ys1 = []
brick_ys2 = []
brick_ys3 = []
brick_ys4 = []

for image_num in range(1, 11):
    # 이미지 읽기
    image_name = './texture_data/train/brick/brick' + str(image_num) + '.jpg'
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    # 이미지에서 영역 짤라내기
    PATCH_SIZE = 32
    
    brick_locations = []
    for i in range(20):
        rand_x = random.randrange(0, image.shape[1])
        rand_y = random.randrange(0, image.shape[0])
        loc_set = (rand_y, rand_x)
        brick_locations.append(loc_set)
    brick_patches = list()
    for loc in brick_locations:
        brick_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                    loc[1]:loc[1] + PATCH_SIZE])
        
    #'contrast', 'dissimilarity', 'homogeneity','energy', 'correlation', 'ASM'
        
    #GLCM dissimilarity와 correlation 계산
# =============================================================================
#     xs = list()
#     ys = list()
#     
#     ys1 = list()
#     ys2 = list()
#     ys3 = list()
# =============================================================================
    
    for patch in (brick_patches):
        glcm = greycomatrix(patch, distances=[1], angles=[0], levels=256,
                            symmetric=False, normed=True)
        # xs.append(greycoprops(glcm, 'dissimilarity')[0,0])
        brick_xs.append(greycoprops(glcm, 'dissimilarity')[0,0])
        # ys.append(greycoprops(glcm, 'correlation')[0,0])
        brick_ys.append(greycoprops(glcm, 'correlation')[0,0])
        brick_ys1.append(greycoprops(glcm, 'contrast')[0,0])
        brick_ys2.append(greycoprops(glcm, 'homogeneity')[0,0])
        brick_ys3.append(greycoprops(glcm, 'energy')[0,0])
        brick_ys4.append(greycoprops(glcm, 'ASM')[0,0])
        
    # #결과 시각화
    # fig = plt.figure(figsize=(8,8))
    
    # ax = fig.add_subplot(3,2,1)
    # ax.imshow(image, cmap=plt.cm.gray, vmin=0, vmax=255)
    # for (y, x) in brick_locations:
    #     ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
    # ax.set_xlabel("Original Image")
    
    # ax = fig.add_subplot(3, 2, 2)
    # ax.plot(xs[:len(brick_patches)], ys[:len(brick_patches)], 'bo')
    # ax.set_xlabel('GLCM Dissimilarity')
    # ax.set_ylabel('GLCM Correlation')
    # ax.legend()
    
    # for i, patch in enumerate(brick_patches):
    #     ax = fig.add_subplot(3, len(brick_patches),
    #                          len(brick_patches)*1 + i + 1)
    #     ax.imshow(patch, cmap=plt.cm.gray,
    #               vmin=0, vmax=255)
    #     ax.set_xlabel('Brick %d' % (i+1))
    
    # plt.tight_layout()
    # plt.show()

print('brick_dissimilarity = ', np.mean(brick_xs))
print('brick_correlation = ', np.mean(brick_ys))
print('brick_contrast = ', np.mean(brick_ys1))
print('brick_homogeneity = ', np.mean(brick_ys2))
print('brick_energy = ', np.mean(brick_ys3))
print('brick_ASM = ', np.mean(brick_ys4))

#====================================

# grass에 대한 train 수행

grass_xs = []
grass_ys = []
grass_con = []
grass_hom = []
grass_enr = []
grass_asm = []

for image_num in range(1, 11):
    # 이미지 읽기
    image_name = './texture_data/train/grass/grass' + str(image_num) + '.jpg'
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    
    # 이미지에서 영역 짤라내기
    PATCH_SIZE = 32
    
    brick_locations = []
    for i in range(20):
        rand_x = random.randrange(0, image.shape[1])
        rand_y = random.randrange(0, image.shape[0])
        loc_set = (rand_y, rand_x)
        brick_locations.append(loc_set)
    brick_patches = list()
    for loc in brick_locations:
        brick_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                    loc[1]:loc[1] + PATCH_SIZE])
        
    #GLCM dissimilarity와 correlation 계산

    for patch in (brick_patches):
        glcm = greycomatrix(patch, distances=[1], angles=[0], levels=256,
                            symmetric=False, normed=True)
        # xs.append(greycoprops(glcm, 'dissimilarity')[0,0])
        grass_xs.append(greycoprops(glcm, 'dissimilarity')[0,0])
        # ys.append(greycoprops(glcm, 'correlation')[0,0])
        grass_ys.append(greycoprops(glcm, 'correlation')[0,0])
        grass_con.append(greycoprops(glcm, 'contrast')[0,0])
        grass_hom.append(greycoprops(glcm, 'homogeneity')[0,0])
        grass_enr.append(greycoprops(glcm, 'energy')[0,0])
        grass_asm.append(greycoprops(glcm, 'ASM')[0,0])
        

print('grass_dissimilarity = ', np.mean(grass_xs))
print('grass_correlation = ', np.mean(grass_ys))
print('grass_contrast = ', np.mean(grass_con))
print('grass_homogeneity = ', np.mean(grass_hom))
print('grass_energy = ', np.mean(grass_enr))
print('grass_ASM = ', np.mean(grass_asm))

#====================================

# grass에 대한 train 수행

ground_xs = []
ground_ys = []
ground_con = []
ground_hom = []
ground_enr = []
ground_asm = []

for image_num in range(1, 11):
    # 이미지 읽기
    image_name = './texture_data/train/ground/ground' + str(image_num) + '.jpg'
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    
    # 이미지에서 영역 짤라내기
    PATCH_SIZE = 32
    
    brick_locations = []
    for i in range(20):
        rand_x = random.randrange(0, image.shape[1])
        rand_y = random.randrange(0, image.shape[0])
        loc_set = (rand_y, rand_x)
        brick_locations.append(loc_set)
    brick_patches = list()
    for loc in brick_locations:
        brick_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                    loc[1]:loc[1] + PATCH_SIZE])
        
    #GLCM dissimilarity와 correlation 계산
    xs = list()
    ys = list()
    for patch in (brick_patches):
        glcm = greycomatrix(patch, distances=[1], angles=[0], levels=256,
                            symmetric=False, normed=True)
        # xs.append(greycoprops(glcm, 'dissimilarity')[0,0])
        ground_xs.append(greycoprops(glcm, 'dissimilarity')[0,0])
        # ys.append(greycoprops(glcm, 'correlation')[0,0])
        ground_ys.append(greycoprops(glcm, 'correlation')[0,0])
        ground_con.append(greycoprops(glcm, 'contrast')[0,0])
        ground_hom.append(greycoprops(glcm, 'homogeneity')[0,0])
        ground_enr.append(greycoprops(glcm, 'energy')[0,0])
        ground_asm.append(greycoprops(glcm, 'ASM')[0,0])

print('ground_dissimilarity = ', np.mean(ground_xs))
print('ground_correlation = ', np.mean(ground_ys))
print('ground_contrast = ', np.mean(ground_con))
print('ground_homogeneity = ', np.mean(ground_hom))
print('ground_energy = ', np.mean(ground_enr))
print('ground_ASM = ', np.mean(ground_asm))

#====================================

# water에 대한 train 수행

water_xs = []
water_ys = []
water_con = []
water_hom = []
water_enr = []
water_asm = []

for image_num in range(1, 11):
    # 이미지 읽기
    image_name = './texture_data/train/water/water' + str(image_num) + '.jpg'
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    
    # 이미지에서 영역 짤라내기
    PATCH_SIZE = 32
    
    brick_locations = []
    for i in range(20):
        rand_x = random.randrange(0, image.shape[1])
        rand_y = random.randrange(0, image.shape[0])
        loc_set = (rand_y, rand_x)
        brick_locations.append(loc_set)
    brick_patches = list()
    for loc in brick_locations:
        brick_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                    loc[1]:loc[1] + PATCH_SIZE])
        
    #GLCM dissimilarity와 correlation 계산
    xs = list()
    ys = list()
    for patch in (brick_patches):
        glcm = greycomatrix(patch, distances=[1], angles=[0], levels=256,
                            symmetric=False, normed=True)
        # xs.append(greycoprops(glcm, 'dissimilarity')[0,0])
        water_xs.append(greycoprops(glcm, 'dissimilarity')[0,0])
        # ys.append(greycoprops(glcm, 'correlation')[0,0])
        water_ys.append(greycoprops(glcm, 'correlation')[0,0])
        water_con.append(greycoprops(glcm, 'contrast')[0,0])
        water_hom.append(greycoprops(glcm, 'homogeneity')[0,0])
        water_enr.append(greycoprops(glcm, 'energy')[0,0])
        water_asm.append(greycoprops(glcm, 'ASM')[0,0])

print('water_dissimilarity = ', np.mean(water_xs))
print('water_correlation = ', np.mean(water_ys))
print('water_contrast = ', np.mean(water_con))
print('water_homogeneity = ', np.mean(water_hom))
print('water_energy = ', np.mean(water_enr))
print('water_ASM = ', np.mean(water_asm))

#====================================

# wood에 대한 train 수행

wood_xs = []
wood_ys = []
wood_con = []
wood_hom = []
wood_enr = []
wood_asm = []

for image_num in range(1, 11):
    # 이미지 읽기
    image_name = './texture_data/train/wood/wood' + str(image_num) + '.jpg'
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    
    # 이미지에서 영역 짤라내기
    PATCH_SIZE = 32
    
    brick_locations = []
    for i in range(20):
        rand_x = random.randrange(0, image.shape[1])
        rand_y = random.randrange(0, image.shape[0])
        loc_set = (rand_y, rand_x)
        brick_locations.append(loc_set)
    brick_patches = list()
    for loc in brick_locations:
        brick_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                    loc[1]:loc[1] + PATCH_SIZE])
        
    #GLCM dissimilarity와 correlation 계산
    xs = list()
    ys = list()
    for patch in (brick_patches):
        glcm = greycomatrix(patch, distances=[1], angles=[0], levels=256,
                            symmetric=False, normed=True)
        # xs.append(greycoprops(glcm, 'dissimilarity')[0,0])
        wood_xs.append(greycoprops(glcm, 'dissimilarity')[0,0])
        # ys.append(greycoprops(glcm, 'correlation')[0,0])
        wood_ys.append(greycoprops(glcm, 'correlation')[0,0])
        wood_con.append(greycoprops(glcm, 'contrast')[0,0])
        wood_hom.append(greycoprops(glcm, 'homogeneity')[0,0])
        wood_enr.append(greycoprops(glcm, 'energy')[0,0])
        wood_asm.append(greycoprops(glcm, 'ASM')[0,0])

print('wood_dissimilarity = ', np.mean(wood_xs))
print('wood_correlation = ', np.mean(wood_ys))
print('wood_contrast = ', np.mean(wood_con))
print('wood_homogeneity = ', np.mean(wood_hom))
print('wood_energy = ', np.mean(wood_enr))
print('wood_ASM = ', np.mean(wood_asm))

#====================================



# =============================================================================
# # 이미지 읽기
# image = cv2.imread('pebbles.jpg',cv2.IMREAD_GRAYSCALE)           # 이미지를 흑백으로 불러오기
# 
# # 이미지에서 풀과 하늘 영역 잘라내기
# PATCH_SIZE = 21                                                 # 이미지에서 잘라낼 영역 너비와 길이 (너비 = 길이 = PATCH_SIZE)
# 
# grass_locations = [(370, 454), (372,22), (444,244), (455,455)]  # 이미지에서 풀 부분 위치 선정, 이미지 좌측 상단이 원점 기준(y축, x축)
# grass_patches = list()
# for loc in grass_locations:
#     grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
#                                loc[1]:loc[1] + PATCH_SIZE])
#     
# sky_locations = [(38,34), (139,28), (37,437), (145, 379)]       # 이미지에서 하늘 부분 위치 선정, 이미지 좌측 상단이 원점 기준(y축, x축)
# sky_patches = list()
# for loc in sky_locations:
#     sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
#                              loc[1]:loc[1] + PATCH_SIZE])
# 
# # 잘라낸 풀과 하늘 영역에서 GLCM dissimilarity와 correlation 계산하기
# xs = list()                                                     # dissimilarity(entropy)값 저장할 list
# ys = list()                                                     # correlation값 저장할 list
# for patch in (grass_patches + sky_patches):
#     glcm = greycomatrix(patch, distances=[1], angles=[0], levels=256,
#                         symmetric=False, normed=True)           # GLCM co-occurence 계산
#     xs.append(greycoprops(glcm, 'dissimilarity')[0,0])          # GLCM dissimilarity(entropy)값 계산하여 xs에 저장
#     ys.append(greycoprops(glcm, 'correlation')[0,0])            # GLCM correlation 계산하여 ys에 저장
#                                                                 # 그외에도 다른 feature 계산 가능
#                                                                 # {'contrast', 'dissimilarity', 'homogeneity',
#                                                                 # 'energy', 'correlation', 'ASM'}
# # 결과 시각화
# fig = plt.figure(figsize=(8,8))                                 # 그림판 만들기
# 
# ax = fig.add_subplot(3,2,1)                                     # 그림판을 3행2열로 나누고 1번 영역에 그림 그리기
# ax.imshow(image, cmap=plt.cm.pray, vmin=0, vmax=255)            # 흑백이미지 추가
# for(y,x) in grass_locations:
#     ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')       # 중심에 초록색 네모(green square)
# for(y,x) in sky_locations:
#     ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')       # 중심에 하늘색 네모
# ax.set_xlabel('Original Image')                                 # x축 이름을 Original Image로
# 
# ax = fig.add_subplot(3,2,2)                                     # 그림판을 3행 2열로 나누고 2번 영역에 그림그리기
# ax.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go', # dissimilarity값을 x축, correlation값을 y축인 지점에 초록색 점 추가
#         label='Grass')
# ax.plot(xs[len(sky_patches):], ys[len(sky_patches):], 'bo',     # dissimilarity값을 x축, correlation값을 y축인 지점에 하늘색 점 추가
#         label='Sky')
# ax.set_xlabel('GLCM Dissimilarity')                             # x축 이름을 GLCM Dissimilarity으로 지정
# ax.set_ylabel('GLCM Correlation')                               # y축 이름을 GLCM Correlation으로 지정
# ax.legend()                                                     # 라벨 이름 추가
# 
# 
# for i, patch in enumerate(grass_patches):                       # 각 풀 영역마다
#     ax = fig.add_subplot(3, len(grass_patches),                 # 그림판을 3행4열로 나누고
#                          len(grass_patches)*1 + i + 1)          # 5,6,7,8번 영역에 그림그리기
#     ax.imshow(patch, cmap=plt.cm.gray,                          # 풀 영역 추가하기
#               vmin=0, vmax=255)                         
#     ax.set_xlabel('Grass %d' %(i + 1))                          # 각 풀 영역 이름 지정
# 
# for i, patch in enumerate(sky_patches):                         # 각 하늘 영역마다
#     ax = fig.add_subplot(3, len(sky_patches),                   # 그림판을 3행4열로 나누고
#                          len(sky_patches)*1 + i + 1)            # 9,10,11,12번 영역에 그림그리기
#     ax.imshow(patch, cmap=plt.cm.gray,                          # 하늘 영역 추가하기
#               vmin=0, vmax=255)                         
#     ax.set_xlabel('Sky %d' %(i + 1))                            # 각 하늘 영역 이름 지정
# 
# plt.tight_layout()                                              # 여백공간 설정
# plt.show()                                                      # 그림 시각화
# 
# 
# =============================================================================
















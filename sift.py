import numpy as np
import cv2 

img1 = cv2.imread('box.png')
img2 = cv2.imread('box_in_scene.png')

sift = cv2.xfeatures2d.SIFT_create()                # sift객체 만들기
kp1, des1 = sift.detectAndCompute(img1,None)        # 첫번째 이미지에서 keypoint와 특징 추출
kp2, des2 = sift.detectAndCompute(img2,None)

bf = cv2.BFMatcher()                                # 점을 둘씩 비교하는 객체
matches = bf.knnMatch(des1, des2, k=2)              # knn 방식으로 비슷한 점 match

good = []
for m, n in matches:
    if m.distance < 0.3*n.distance:                 # 다른점들보다 많이 비슷하면 
        good.append([m])
        
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)    #두개의 특징점을 서로 연결
cv2.imshow('image', img3)
cv2.waitKey()
cv2.destropAllWindows()
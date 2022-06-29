import numpy as np
import cv2 

image = cv2.imread('building.jpg', cv2.IMREAD_GRAYSCALE)

harris = cv2.cornerHarris(image, blockSize=3, ksize=3, k=0.04)          # harris corner R값 계산

harris_norm = cv2.normalize(harris, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)   # 0~255로 정규화

image2 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

for y in range(1, harris_norm.shape[0]-1):
    for x in range(1, harris_norm.shape[1]-1):
        if harris_norm[y,x] > 120:
            if (harris[y,x] > harris[y-1, x] and
                harris[y,x] > harris[y+1, x] and
                harris[y,x] > harris[y, x-1] and
                harris[y,x] > harris[y, x+1]):
                cv2.circle(image2, (x,y), radius=5, color=(0,0,255),
                           thickness=2)                                     # 빨간색 동그라미로 코너 표시
                    
cv2.imshow('image',image)                                                   # 흑백이미지 출력
cv2.imshow('harris_norm', harris_norm)                                      # harris 계산값 출력
cv2.imshow('output', image2)                                                # 코너 검출 값 출력
cv2.waitKey()
cv2.destropAllWindows()

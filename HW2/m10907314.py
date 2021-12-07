import numpy as np
import cv2

image_src = cv2.imread("./ArtGallery.jpg")

point_left = np.array([[40,103],[342,122],[342,427],[43,447]])
point_right = np.array([[684,119],[957,69],[955,517],[680,454]])

#計算轉換矩陣
left_to_right_h,status = cv2.findHomography(point_left,point_right)
right_to_left_h,status = cv2.findHomography(point_right,point_left)

#左右幅畫四個頂點所涵蓋的所有範圍(二值化灰階圖)
left_mask = np.zeros(image_src.shape[:2], np.uint8)
cv2.drawContours(left_mask, [point_left], -1, (255, 255, 255), -1, cv2.LINE_AA)
cv2.imshow("left_mask.png", left_mask)

right_mask = np.zeros(image_src.shape[:2], np.uint8)
cv2.drawContours(right_mask, [point_right], -1, (255, 255, 255), -1, cv2.LINE_AA)
cv2.imshow("right_mask.png", right_mask)

#擷取出左右幅畫
left_remain = cv2.bitwise_and(image_src, image_src, mask=left_mask)
cv2.imshow("left_remain.png", left_remain)

right_remain = cv2.bitwise_and(image_src, image_src, mask=right_mask)
cv2.imshow("right_remain.png", right_remain)

#將圖片進行轉換 左邊幅畫右轉到右邊 右邊幅畫轉到左邊
left_to_right_image = cv2.warpPerspective(left_remain,left_to_right_h,(image_src.shape[1],image_src.shape[0]))
right_to_left_image = cv2.warpPerspective(right_remain,right_to_left_h,(image_src.shape[1],image_src.shape[0]))
cv2.imshow("left_to_right_image.png", left_to_right_image)
cv2.imshow("right_to_left_image.png", right_to_left_image)

#將轉換過後的圖片做灰階二值 就能知道涵蓋範圍
Grayimg_left = cv2.cvtColor(left_to_right_image, cv2.COLOR_BGR2GRAY)
ret_left, thresh_left = cv2.threshold(Grayimg_left, 1, 255,cv2.THRESH_BINARY)
cv2.imshow("thresh_left.png", thresh_left)

Grayimg_right = cv2.cvtColor(right_to_left_image, cv2.COLOR_BGR2GRAY)
ret_right, thresh_right = cv2.threshold(Grayimg_right, 1, 255,cv2.THRESH_BINARY)
cv2.imshow("thresh_right.png", thresh_right)

#將原圖移去幅畫涵蓋範圍
image_mask = cv2.add(image_src, np.zeros(np.shape(image_src), dtype=np.uint8), mask=~thresh_left)
image_mask = cv2.add(image_mask, np.zeros(np.shape(image_src), dtype=np.uint8), mask=~thresh_right)
cv2.imshow("image_mask",image_mask)

#將三張圖片合併得到結果
result_image = image_mask + left_to_right_image + right_to_left_image

#Display images;
cv2.imshow("Source image",image_src)
cv2.imshow("result_image",result_image)
cv2.imwrite("m10907314.jpg",result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
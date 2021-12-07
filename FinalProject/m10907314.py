import numpy as np
import cv2
import glob

#Left Camera K (3x3)
LeftK = np.array([[1496.880651, 0.000000, 605.175810],
                    [0.000000, 1490.679493, 338.418796],
                    [0.000000, 0.000000, 1.000000]])

#Left Camera RT (3x4)
LeftRT = np.array([[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0]])

#Right Camera K (3x3)
RightK = np.array([[1484.936861, 0.000000, 625.964760],
                    [0.000000, 1480.722847, 357.750205],
                    [0.000000, 0.000000, 1.000000]])

#Right Camera RT (3x4)
RightRT = np.array([[0.893946, 0.004543, 0.448151, -186.807456],
                    [0.013206, 0.999247, -0.036473, 3.343985],
                    [-0.447979, 0.038523, 0.893214, 45.030463]])

#Fundamental Matrix (3x3)
F21 = np.array([[0.000000191234, 0.000003409602, -0.001899934537],
                [0.000003427498, -0.000000298416, -0.023839273818], 
                [-0.000612047140, 0.019636148869, 1.000000000000]])


#將兩張照片的特徵點透過Fundamental matrix進行匹配
def Match(LeftFeature, RightFeature, F21):
    #補1在最後一行 [u v 1]
    one = np.ones(LeftFeature.shape[0])
    LeftFeature = np.column_stack( (LeftFeature, one) )
    one = np.ones(RightFeature.shape[0])
    RightFeature = np.column_stack( (RightFeature, one) )
    
    #進行xFx的運算
    result = RightFeature.dot(F21.dot(LeftFeature.transpose()))

    #任一左圖特徵點在所有右圖點中找出距離誤差最小的
    min_difference_index = np.argmin(abs(result), axis=0)
    min_difference = np.min(abs(result), axis=0)

    #儲存特徵點的index，距離誤差要小於0.5才會算匹配
    match = np.empty([0,2])
    for i in range(len(min_difference)):
        if min_difference[i] < 0.5:
            match = np.row_stack((match,[i,min_difference_index[i]]))
    return match

#找出有光的那條線上的特徵點
def FeatureExtract(img):
    feature = np.empty([0,2], dtype='int')

    ret,thresh1 = cv2.threshold(img,70,255,cv2.THRESH_BINARY)  #對相片先進行二值化找出亮度大於70的點
    row, col= (thresh1 > 0).nonzero() #取出二值化圖片中不為0的所有點
    
    #將同一行上的有亮的點的位置做平均得到subpixel
    previous_row = -1
    now_row = -1
    count = 0
    sum_col = 0
    for i in range(len(row)):
        now_row = row[i]
        if previous_row != -1 and now_row != previous_row:
            average = sum_col/count
            feature = np.row_stack((feature,[average,previous_row]))
            count = 0
            sum_col = 0
        count += 1
        sum_col += col[i]
        previous_row = now_row
    if len(row) > 0:
        average = sum_col/count
        feature = np.row_stack((feature,[average,previous_row]))

    #print('feature count: ', len(feature))
    return feature
    
#透過兩張照片的像素點以及投影矩陣 就能透過這三角化函數求出三維座標點
def Triangulation(x1, x2, P1, P2):
    #將公式中的各個項目填入矩陣A中
    u1 = x1[0]
    v1 = x1[1]
    u2 = x2[0]
    v2 = x2[1]
    A = np.array([u1*P1[2]-P1[0],
                  v1*P1[2]-P1[1],
                  u2*P2[2]-P2[0],
                  v2*P2[2]-P2[1]], dtype='float32')
    
    #SVD求解後取V的最後一行為最小二乘解
    U,sigma,VT=np.linalg.svd(A)
    V = VT.transpose()
    X = V[:, -1]
    X = X / X[3]
    return X

#將三維點透過投影矩陣投影回像素與觀測到的像素點做比較誤差
def Reproject_Error(observe_x, P, X):
    reproject_x = P.dot(X)
    reproject_x = reproject_x / reproject_x[2]
    observe_x[0] -= reproject_x[0]
    observe_x[1] -= reproject_x[1]
    return np.linalg.norm(observe_x)

#將資料夾中的照片一一取出執行
print("start")
index = 0
f = open( 'm10907314.xyz', 'w')
imagelist= sorted(glob.glob('./SidebySide/' + 'SBS_*.jpg'))
for path in imagelist:
    index = index+1
    #讀入灰階圖片後高斯模糊，為了抑制雜訊誤差
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.blur(img, (5, 5))
    
    #分為左右兩張照片
    LeftImg = img[:, :int(img.shape[1]/2)]
    RightImg = img[:, int(img.shape[1]/2):]
    
    #個別取出照片中的特徵點
    LeftFeature = FeatureExtract(LeftImg)
    RightFeature = FeatureExtract(RightImg)
    
    #進行特徵匹配
    match = Match(LeftFeature, RightFeature, F21)
    
    #算出左右相機的投影矩陣
    P1 = np.dot(LeftK, LeftRT)
    P2 = np.dot(RightK, RightRT)
    
    #對所有匹配的點做三角化，若重投影誤差大於1則不採用
    for i in range(len(match)):
        x1 = LeftFeature[ int(match[i][0]) ]
        x2 = RightFeature[ int(match[i][1]) ]
        X = Triangulation(x1, x2, P1, P2)
        if Reproject_Error(x1, P1, X) > 1 or Reproject_Error(x2, P2, X) > 1:
            continue
        f.write(str(X[0]) + ' ' + str(X[1]) + ' ' + str(X[2]) + '\n')
        
    print('index: ', index , ' finish')

f.close()
print("finish")

# cv2.waitKey(0)
# cv2.destroyAllWindows()


import numpy as np
import cv2
import sys

#內參和外參矩陣
Instrisic = np.array([[722.481995, 0.000000, 399.000000],
                    [0.000000, 722.481934, 311.000000],
                    [0.000000, 0.000000, 1.000000]])

Extrinsic = np.array([[[0.508468, 0.860935, -0.015857, 3.932282],
                [-0.101237, 0.041482, -0.994003, 21.303495],
                [-0.855111, 0.507022, 0.108232, 88.085114]],
               
               [[0.054555, 0.997525, 0.044349, -0.396882],
                [-0.129876, 0.051127, -0.990217, 21.030863],
                [-0.990031, 0.048262, 0.132325, 86.551529]],

                [[-0.483178, 0.864219, 0.140232, -6.133193],
                [-0.118992, 0.093863, -0.988454, 21.073915],
                [-0.867401, -0.494281, 0.057464, 88.905014]],
                
                [[-0.826707, 0.562054, 0.025509, -3.706015],
                [-0.015798, 0.022133, -0.999636, 22.618803],
                [-0.562411, -0.826804, -0.009437, 89.343170]],
                
                [[-0.920328, -0.386141, 0.062375, 0.837949],
                [-0.102550, 0.084312, -0.991155, 20.728113],
                [0.377463, -0.918577, -0.117211, 93.253716]],
                
                [[-0.315707, -0.948836, -0.006217, 2.520231],
                [-0.091609, 0.037001, -0.995114, 20.983803],
                [0.944422, -0.313592, -0.098622, 93.845718]],
                
                [[0.547250, -0.836131, -0.037448, -5.300584],
                [-0.082565, -0.009407, -0.996548, 21.209290],
                [0.832886, 0.548449, -0.074202, 91.918419]],

                [[0.998758, -0.047722, -0.014321, 7.249783],
                [-0.026645, -0.268707, -0.962861, 16.707335],
                [0.042101, 0.962035, -0.269661, 101.258018]],
                
                [[0.515633, 0.856808, 0.001705, 1.082603],
                [0.856751, -0.515573, -0.013213, 4.623665],
                [-0.010425, 0.008264, -0.999911, 114.083206]],
                
                [[-0.164361, 0.974144, -0.155012, 2.258266],
                [-0.915024, -0.209267, -0.344892, 8.572358],
                [-0.368429, 0.085148, 0.925748, 76.273094]],
                
                [[-0.053241, 0.998369, 0.020604, 1.405633],
                [-0.737301, -0.053217, 0.673473, -20.414452],
                [0.673454, 0.020664, 0.738940, 80.795265]]])

path_all = ["Bird", "Last", "Monkey", "Teapot"] #所有的路徑
point3D = np.arange(4000000).reshape(4,1000000) #先建立一個4x1000000的3D點(x,y,z,1)矩陣

#將對應的3D點座標寫入矩陣(-50,-50,0) to (50,50,100)
index = 0
for x in range(100):
    for y in range(100):
        for z in range(100):
            point3D[0][index] = x-50;
            point3D[1][index] = y-50;
            point3D[2][index] = z;
            point3D[3][index] = 1;
            index = index + 1

for path in path_all:  #for迴圈所有要做的路徑
    voxel = np.ones(1000000)  #將所有3D點的voxel設為1 假設所有點都要寫入
    for index in range(11):  #將11張照片都處理
        bmp = cv2.imread("./" + path + "/" + str(index+1).zfill(2) + ".bmp") #讀入照片
        print(path + str(index+1).zfill(2) + ".bmp")
    
        #內參乘上外參再乘上所有的3D點矩陣得到對應到相機上的pixel點
        #將最後一個值強制轉為1後轉成int型態
        KRt = Instrisic.dot(Extrinsic[index])
        pixel2D = KRt.dot(point3D)
        pixel2D[0] = pixel2D[0]/pixel2D[2]
        pixel2D[1] = pixel2D[1]/pixel2D[2]
        pixel2D[2] = pixel2D[2]/pixel2D[2]
        pixel2D = pixel2D.astype('int')
    
        
        
        for i in range(1000000):
            if voxel[i] == 1:
                
                #如果超出圖片大小範圍 移除voxel
                if int(pixel2D[1][i]) >= bmp.shape[0] or int(pixel2D[1][i]) <0 or int(pixel2D[0][i]) >= bmp.shape[1] or int(pixel2D[0][i]) <0:
                    voxel[i] = 0
                    continue
                
                #如果在圖片背景 移除voxel
                if bmp[int(pixel2D[1][i])][int(pixel2D[0][i])][0] == 0:
                    voxel[i] = 0

                    
    #將voxel為1的3D點寫入xyz檔中
    print("writing file")
    f = open( path +'.xyz', 'w')
    index = 0
    for x in range(100):
        for y in range(100):
            for z in range(100):
                if voxel[index] == 1:
                    f.write(str(x-50) + " " + str(y-50) + " " + str(z) + "\n")
                index = index + 1
    f.close()
    print("finish")


# for index in range(11):
#     bmp = cv2.imread(path + str(index+1).zfill(2) + ".bmp")
#     print(path + str(index+1).zfill(2) + ".bmp")
#     # print(path + str(index+1).zfill(2) + ".bmp")
#     # print(bmp[4][-626][0])
#     # break
#     KRt = Instrisic.dot(Extrinsic[index])

#     for x in range(100):
#         for y in range(100):
#             for z in range(100):
#                 if voxel[x][y][z] == 1:
#                     pixel = KRt.dot(np.array([x-50, y-50, z, 1]))
#                     # pixel /= pixel[2]
#                     # print(pixel)
#                     pixel_x = int(pixel[0] / pixel[2])
#                     pixel_y = int(pixel[1] / pixel[2])
                    
#                     if pixel_y >= bmp.shape[0] or pixel_y <0 or pixel_x >= bmp.shape[1] or pixel_x <0:
#                         voxel[x][y][z] = 0
#                         continue
#                     print("pixel_x:" + str(pixel_x) + " pixel_y:" + str(pixel_y))
#                     if bmp[pixel_y][pixel_x][0] == 0:
#                         voxel[x][y][z] = 0
                        
# f = open('Bird.xyz', 'w')
# for x in range(100):
#     for y in range(100):
#         for z in range(100):
#             if voxel[x][y][z] == 1:
#                 f.write(str(x-50) + " " + str(y-50) + " " + str(z) + "\n")
# f.close()


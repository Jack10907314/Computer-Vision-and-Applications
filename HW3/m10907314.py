import numpy as np
import cv2

def GoldStandard(points):
    n = points.shape[1]
    #先去掉最後一行的1
    standard_points = np.delete(points, n-1, 1)
    
    #求出質心後將所有座標點偏移質心座標
    centroid = np.mean(standard_points, axis=0)
    standard_points = standard_points - centroid
    
    #求出所有點的平均距離L，scale = 根號2 / L 
    average_distance = np.average( np.linalg.norm(standard_points, axis=1)) 
    s = np.power(2,1/2) / average_distance
    standard_points = standard_points * s
    
    #填入ST矩陣
    ST = np.zeros((n,n))
    for i in range(n-1):
        ST[i][i] = s
    for i in range(n-1):
        ST[i][n-1] = -1*s*centroid[i]
    ST[n-1][n-1] = 1
    
    #將最後一行1補回去
    one = np.ones(standard_points.shape[0])
    standard_points = np.column_stack( (standard_points, one) )

    return ST, standard_points

#相機內參
K = np.array([[1308.36, 0.00, 780.00],
             [0.00, 1308.36, 480.50],
             [0.00, 0.00, 1.00]])


#6點相機座標
camera_2D = np.array([[228, 416, 1],
                      [671, 294, 1],
                      [303, 842, 1],
                      [1148, 596, 1],
                      [1370, 706, 1],
                      [1108, 772, 1]])

#6點世界座標
world_3D = np.array([[-100, 50, 100, 1],
                    [0, 50, 100, 1],
                    [-100, 50, 0, 1],
                    [50, -50, 50, 1],
                    [50, -100, 50, 1],
                    [50, -50, 0, 1]])

#目標量測點
target_3D = [-4.5, -2.5, 130.0, 1]


if __name__ == '__main__':
    
    #先將所有點做normalization
    camera_ST, camera_2D = GoldStandard(camera_2D)
    world_ST, world_3D = GoldStandard(world_3D)
    print("camera_ST:\n", camera_ST, "\n")
    print("world_ST:\n", world_ST, "\n")
    
    #將數值填入A矩陣
    number_of_points = camera_2D.shape[0]
    A = np.zeros( (number_of_points*2, 12) )
    for i in range(number_of_points):
        A[i*2][0] = world_3D[i][0]
        A[i*2][1] = world_3D[i][1]
        A[i*2][2] = world_3D[i][2]
        A[i*2][3] = world_3D[i][3]
        A[i*2][4] = 0
        A[i*2][5] = 0
        A[i*2][6] = 0
        A[i*2][7] = 0
        A[i*2][8] = -1 * camera_2D[i][0] * world_3D[i][0]
        A[i*2][9] = -1 * camera_2D[i][0] * world_3D[i][1]
        A[i*2][10] = -1 * camera_2D[i][0] * world_3D[i][2]
        A[i*2][11] = -1 * camera_2D[i][0] * world_3D[i][3]

        A[i*2+1][0] = 0
        A[i*2+1][1] = 0
        A[i*2+1][2] = 0
        A[i*2+1][3] = 0
        A[i*2+1][4] = world_3D[i][0]
        A[i*2+1][5] = world_3D[i][1]
        A[i*2+1][6] = world_3D[i][2]
        A[i*2+1][7] = world_3D[i][3]
        A[i*2+1][8] = -1 * camera_2D[i][1] * world_3D[i][0]
        A[i*2+1][9] = -1 * camera_2D[i][1] * world_3D[i][1]
        A[i*2+1][10] = -1 * camera_2D[i][1] * world_3D[i][2]
        A[i*2+1][11] = -1 * camera_2D[i][1] * world_3D[i][3]
    print("A.shape: \n",A.shape)
    print("A:\n", A, "\n")
    
    #SVD求解A矩陣
    U,sigma,VT=np.linalg.svd(A)
    V = VT.transpose()
    #取V的最後一行
    standard_P = np.array([[V[0][11], V[1][11], V[2][11], V[3][11]],
                          [V[4][11], V[5][11], V[6][11], V[7][11]],
                          [V[8][11], V[9][11], V[10][11], V[11][11]]])
    #正規化的投影矩陣 轉成 原本的投影矩陣，並scale
    P = np.dot( np.linalg.inv(camera_ST), standard_P )
    P = np.dot( P, world_ST)
    P = P / P[2][3]
    print("P:\n", P, "\n")
    
    #求出Rt後，取第三行做scale
    Rt = np.dot(np.linalg.inv(K), P)
    scale = np.linalg.norm(np.array([Rt[0][2], Rt[1][2], Rt[2][2]]))
    Rt = Rt / scale
    print("Rt:\n", Rt, "\n")
    
    #查看R的正交特性
    a = np.array([Rt[0][0], Rt[1][0], Rt[2][0]])
    b = np.array([Rt[0][1], Rt[1][1], Rt[2][1]])
    c = np.array([Rt[0][2], Rt[1][2], Rt[2][2]])
    print("a dot b = ", np.dot(a, b))
    print("b dot c = ", np.dot(b, c))
    print("c dot a = ", np.dot(c, a))
    print()
    
    R = np.delete(Rt, 3, 1)
    print("RR^T:\n", R.dot(R.transpose()), "\n")

    #Rt內積目標世界座標位置轉換成相機座標系位置，得到答案
    target_vector = Rt.dot(target_3D)
    print("target vector:\n", target_vector, "\n")
    
    print("distance: ", np.linalg.norm(target_vector))
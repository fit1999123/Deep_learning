import numpy as np
import matplotlib.pyplot as plt 
import cv2


flat_chess = cv2.imread("chess_board.jpg")
flat_chess = cv2.resize(flat_chess,(350,350))
real_chess = cv2.imread("chess.jpg")
real_chess = cv2.resize(real_chess,(600,400))

flat_chess = cv2.cvtColor(flat_chess,cv2.COLOR_BGR2RGB)
flat_chess2 = np.copy(flat_chess)
gray_flat_chess = cv2.cvtColor(flat_chess,cv2.COLOR_RGB2GRAY)
real_chess = cv2.cvtColor(real_chess,cv2.COLOR_BGR2RGB)
real_chess2 = np.copy(real_chess)

gray_real_chess = cv2.cvtColor(real_chess,cv2.COLOR_RGB2GRAY)

gray = np.float32(gray_flat_chess)
gray2 = np.float32(gray_real_chess)



dst = cv2.cornerHarris(src = gray,blockSize=2,ksize=3,k=0.04)
dst = cv2.dilate(dst,None)

dst2 = cv2.cornerHarris(src = gray2,blockSize=2,ksize=3,k=0.06)
dst2 = cv2.dilate(dst2,None)

flat_chess[dst>0.01*dst.max()] = [255,0,0]
real_chess[dst2>0.01*dst2.max()] = [255,0,0]

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(flat_chess)
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(real_chess)

coners = cv2.goodFeaturesToTrack(gray_flat_chess,100,0.01,10)
coners = np.int0(coners)
coners2 = cv2.goodFeaturesToTrack(gray_real_chess,100,0.01,10)
coners2 = np.int0(coners2)


# print(coners)

for i in coners:
    x,y = i.ravel()
    cv2.circle(flat_chess2,(x,y),3,(255,0,0),-1)
for i in coners2:
    x,y = i.ravel()
    cv2.circle(real_chess2,(x,y),3,(255,0,0),-1)


ax3 = fig.add_subplot(2,2,3)
ax3.imshow(flat_chess2)
ax4 = fig.add_subplot(2,2,4)
ax4.imshow(real_chess2)


plt.show()
plt.close()
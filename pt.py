import numpy as np
import matplotlib.pyplot as plt
import cv2

# 基準とする畳四隅の写真上の座標（単位px）
pts1 = np.array([(104, 202), (342, 244), (304, 495), (85, 452)], dtype=np.float32)
# 基準とする畳四隅の実際の座標（単位cm）
pts2 = np.array([(50,50), (250,50), (250,250), (50,250)], dtype=np.float32)

# 射影行列の取得
M = cv2.getPerspectiveTransform(pts1, pts2) 
np.set_printoptions(precision=5, suppress=True)
print (M)

# 元画像  (W x H) = (2016 x 1512)
img1 = cv2.imread('Blurqr.png', cv2.IMREAD_COLOR)
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

# 元画像を射影変換し鳥瞰画像へ
w2, h2 = pts2.max(axis=0).astype(int) + 50 # 鳥瞰画像サイズを拡張（見た目の調整） 
img2 = cv2.warpPerspective(img1, M, (w2,h2) )
cv2.imwrite('Blurqrsquare.jpg', img2)

# 結果表示 
fig = plt.figure(figsize=(8,8))
fig.add_subplot(1,2,1).imshow(img1)
fig.add_subplot(1,2,2).imshow(img2)
plt.show()
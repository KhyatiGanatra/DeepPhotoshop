
import sys
import cv2

img = cv2.imread(sys.argv[1])
img = cv2.resize(img, (512, 512), cv2.INTER_AREA)

cv2.imwrite(sys.argv[1], img)



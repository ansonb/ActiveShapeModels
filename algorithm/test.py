from asm import ASM
from scipy import misc
import cv2

asm = ASM()

test_img = misc.imread('./../data/prepared_data/test/subject04.surprised.jpg')

features = asm.asm(test_img)
print(features)
for i in range(len(features)//2):
	cv2.circle(test_img, (int(abs(features[i])), int(abs(features[i+len(features)//2]))), 1, (255,0,0), -1)
cv2.imwrite('result.jpg',test_img)
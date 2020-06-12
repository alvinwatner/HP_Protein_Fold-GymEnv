import numpy as np 

def create_init_image():
	img = np.zeros((6000, 6000, 3), dtype=np.uint8)
	for i in range(len(img)):
		for j in range(len(img)):
			img[i][j][0] = 155
			img[i][j][1] = 175
	return img


im = create_init_image()
np.save("Background_Img/background.npy", im)


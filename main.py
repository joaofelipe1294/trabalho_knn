from base import BaseLoader
from knn import KNN
import cv2
from PIL import Image
import numpy as np
from pre_processor import PreProcessor
import time

#X_train, y_train = BaseLoader().load_text_values('150k/CCtrain')
#X_test, y_test = BaseLoader().load_text_values('150k/CCtest3')
#knn = KNN(X_train, y_train, X_test, y_test)
#knn.predict(5)

#data = cv2.imread('digits/train/cdf0000_14_3_7.tif', 0)
#img = Image.fromarray(data) #.convert('LA')
#img = img.resize((data.shape[0] // 8, data.shape[1] // 8), Image.ANTIALIAS)
#img.show()
#image = np.array(img)
#print(image.shape)
#imgplot = plt.imshow(img)

start_time = time.time()
X_train, y_train = BaseLoader().load_images('digits/train', 'digits/train.txt')[:2]
#X_train = PreProcessor().re_size_images(X_train, 8)
X_test, y_test = BaseLoader().load_images('digits/test', 'digits/test.txt')[:2]
pre_processor = PreProcessor()
#X_test = PreProcessor().re_size_images(X_test, 8)
#knn = KNN(X_train, y_train, X_test, y_test)
#knn.predict(5)

X_train, y_train = pre_processor.calc_mean_images(X_train, y_train)
X_train = pre_processor.prepare_image_values(X_train)
X_test = pre_processor.prepare_image_values(X_test)

knn = KNN(X_train, y_train, X_test, y_test)
knn.predict(1)


end_time = time.time() - start_time
print('Concluido em : %f' % end_time)

#for index in range(0, len(template_labels)):
#	img = Image.fromarray(templates[index]) #.convert('LA')
#	img.show(title = str(index))

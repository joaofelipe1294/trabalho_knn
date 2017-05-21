from base import BaseLoader
from knn import KNN
import cv2
from PIL import Image
import numpy as np
from pre_processor import PreProcessor
import time
import sys

data_type = sys.argv[1]
train_path = sys.argv[2]
test_path = sys.argv[3]
K = int(sys.argv[4])

if data_type == '-t':
	print('Texto')
	if len(sys.argv) == 7:
		train_lines = int(sys.argv[5])
		test_lines = int(sys.argv[6])
		X_train, y_train = BaseLoader().load_text_values(train_path, lines_number = train_lines)
		X_test, y_test = BaseLoader().load_text_values(test_path, lines_number = test_lines)
	else:
		X_train, y_train = BaseLoader().load_text_values(train_path)
		X_test, y_test = BaseLoader().load_text_values(test_path)
elif data_type == '-i':
	print('Imagens')

knn = KNN(X_train, y_train, X_test, y_test)
knn.predict(K)




'''
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
'''
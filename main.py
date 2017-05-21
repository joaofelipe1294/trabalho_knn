from base import BaseLoader
from knn import KNN
from PIL import Image
import numpy as np
from pre_processor import PreProcessor
import time
import sys

data_type = sys.argv[1]
train_path = sys.argv[2]
test_path = sys.argv[3]
K = int(sys.argv[4])

base = BaseLoader()
start_time = time.time()
if data_type == '-t':
	print('Texto')
	if len(sys.argv) == 7:
		train_lines = int(sys.argv[5])
		test_lines = int(sys.argv[6])
		X_train, y_train = base.load_text_values(train_path, lines_number = train_lines)
		X_test, y_test = base.load_text_values(test_path, lines_number = test_lines)
	else:
		X_train, y_train = base.load_text_values(train_path)
		X_test, y_test = base.load_text_values(test_path)
elif data_type == '-i':
	print('Imagens')
	train_file = train_path + '.txt'
	test_file = test_path + '.txt'
	if len(sys.argv) == 7:
		print('Particionado')
		train_number = int(sys.argv[5])
		test_number = int(sys.argv[6])
		X_train, y_train = base.load_images(train_path, train_file, samples_number = train_number)[:2]
		X_test, y_test = base.load_images(test_path, test_file, samples_number = test_number)[:2]
	else:
		X_train, y_train = base.load_images(train_path, train_file)[:2]
		X_test, y_test = base.load_images(test_path, test_file)[:2]
	print('+------------------+---------------+------------+')
	print('|1 - Redimensionar | 2 - Templates | 3 - Normal |')
	print('+------------------+---------------+------------+')
	pre_process_type = int(input(': '))
	pre_processor = PreProcessor()
	if pre_process_type == 1:
		new_size = int(input('Dividir por : '))
		X_train = pre_processor.re_size_images(X_train, new_size)
		X_test = pre_processor.re_size_images(X_test, new_size)
	elif pre_process_type == 2:
		X_train, y_train = pre_processor.calc_mean_images(X_train, y_train)
		X_test = pre_processor.prepare_image_values(X_test)
	elif pre_process_type == 3:
		X_train = pre_processor.prepare_image_values(X_train)
		X_test = pre_processor.prepare_image_values(X_test)
	print('Pre-processamento concluido')



knn = KNN(X_train, y_train, X_test, y_test)
knn.predict(K)
print('Concluido em : %f' % (time.time() - start_time))

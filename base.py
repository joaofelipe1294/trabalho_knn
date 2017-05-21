# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2


class BaseLoader:

	def load_images(self, base_path, labels_file_path, divisor = ' '):
		#metodo utilizado para carregar os valores da base de imagens
		print("Carregando imagens de " + base_path + " ...")
		names_txt = []
		labels_txt = []
		with open(labels_file_path, 'r') as file:  #pega as labels do arquivo txt da base de digitos 
			lines = file.readlines()
			for line in lines:
				values = line.split(' ')
				names_txt.append(values[0])
				labels_txt.append(int(values[1]))
		images = []
		labels = []
		names = []
		paths = os.listdir(base_path)  #pega os paths das imagens da base
		paths.sort()                   #ordena a lista com os paths por nome 
		for path in paths:             #loop que itera sobre todos os paths de imagem
			image_path = base_path + '/' + path       #prepara o path para leitura da imagem
			label = labels_txt[names_txt.index(path)] #atribui a label
			images.append(cv2.imread(image_path, 0))  #le a imagem
			labels.append(label)
			names.append(path)
		print("Imagens carregadas")
		return images, labels, names
		
	def load_text_values(self, base_path, lines_number = None, divisor = ' '):
		#metodo utilizado para carregar os valores de uma base do tipo txt
		print('iniciando leitura arquivo')
		X = [] #lista com os valores
		y = [] #lista com as labels
		with open(base_path, 'r') as file:
			file.readline()
			lines = file.readlines()
			for line in lines:
				values = line.split(divisor) #divide o arquivo pelo caractere X
				[float(i) for i in values] #converte valores para float
				y.append(int(values.pop(len(values) - 1))) #isola o valor referente a label
				X.append(values)	
		if lines_number:
			y = y[:lines_number]
			X = X[:lines_number]
		X = np.array( X, np.float32) #retorna um np.array do tipo float porque os valores da base 150k sao do tipo fload
		return X, y

		

#images, labels, names = BaseLoader().load_images('digits/train', 'digits/train.txt')
#values, labels = BaseLoader().load_text_values('150k/CCtrain')


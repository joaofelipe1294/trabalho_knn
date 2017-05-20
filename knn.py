import numpy as np
from collections import defaultdict
from collections import Counter


class KNN:

	def __init__(self, X_train, y_train, X_test, y_test):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test
		self.predicted_labels = []

	def predict(self, K):
		for index in range(0, self.X_test.shape[0]):
			sample = self.X_test[index, :]
			copy_matrix = np.full(self.X_train.shape, sample, np.float32) #criando matriz com valores do novo ponto
			distances_matrix = (self.X_train - copy_matrix) ** 2 #calcula a distancia dos pontos de treino com o de teste
			distance_values = distances_matrix.sum(axis = 1) #computa a soma dos valore de cada uma das linhas, retorna um vetor
			count = 0
			min_indexes = [] #lita com o indice dos pontos com menor distancia
			while count < K: #loop que descobre os indices que possuem as menores distancias
				index = np.argmin(distance_values) #recupera o indice do menor valor
				min_indexes.append(index) #adiciona o indice na lista que possui o menor vetor 
				distance_values.itemset(index, False) #adiciona valor sentinela para que nao seja contado novamente
				count += 1
			final_labels = [] #lista com a label referente aos pontos com menor distancia
			for index in min_indexes: 
				final_labels.append(self.y_train[index])
			scores = Counter(final_labels)
			label = scores.most_common(1)[0][0]
			self.predicted_labels.append(label)
		self.calc_precision()
		self.calc_confusion_matrix()

	def calc_precision(self):
		corrects = 0
		for index in range(0, len(self.y_test)):
			if self.y_test[index] == self.predicted_labels[index]:
				corrects += 1
		precision = float(corrects) / float(len(self.y_test))
		print('Precision : %f' % precision)

	def calc_confusion_matrix(self):
		confusion_matrix = np.zeros((10,10), np.uint32)
		for index in range(0, len(self.y_test)):
			confusion_matrix[self.y_test[index], self.predicted_labels[index]] += 1
		print(confusion_matrix)

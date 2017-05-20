import numpy as np
from PIL import Image


class PreProcessor:

	def re_size_images(self, images, size):
		re_sized_images = []
		for image in images:
			img = Image.fromarray(image)
			img = img.resize((image.shape[0] // size, image.shape[1] // size), Image.ANTIALIAS)
			re_sized_image = np.array(img)
			re_sized_images.append(re_sized_image)
		values = self.prepare_image_values(re_sized_images)
		return values

	def prepare_image_values(self, values):
		prepared_values = []
		for image in values:
			prepared_values.append(np.ravel(image))
		prepared_values = np.array(prepared_values, np.uint8)
		return prepared_values

	def calc_mean_images(self, images, labels):
		templates = []
		template_labels = []
		templates_number = int(input('Templates por classe : '))
		clusters = [[], [], [], [] , [] , [] , [] , [] , [] , []]
		for index in range(0, len(labels)):
			clusters[labels[index]].append(images[index])
		images_by_template = int(len(clusters[0]) / templates_number)
		for label in range(0, 10):
			for template in range(0, templates_number):
				avrage_image = np.zeros(images[0].shape, np.uint64)
				for index in range(template * images_by_template , (template + 1) * images_by_template):
					avrage_image += clusters[label][index]
				avrage_image = avrage_image / images_by_template
				avrage_image = np.array(avrage_image, np.uint8)
				templates.append(avrage_image)
				template_labels.append(label)
		return templates, template_labels
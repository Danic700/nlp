import math
import sys
from collections import Counter


class KNNClassifier():

	def _get_cosine(self, vec1, vec2):
		intersection = set(vec1.keys()) & set(vec2.keys())
		numerator = sum([vec1[x] * vec2[x] for x in intersection])

		sum1 = sum([vec1[x]**2 for x in vec1.keys()])
		sum2 = sum([vec2[x]**2 for x in vec2.keys()])
		denominator = math.sqrt(sum1) * math.sqrt(sum2)

		if not denominator:
			return 0.0
		else:
			return float(numerator) / denominator

	def classify_all(self):
		new_data = []
		for doc in self.data:
			# [word_vec, category]
			new_data += [[Counter(doc[0] + doc[1]), doc[2]]]

		success = 0
		error = 0

		for i in range(len(new_data)):

			max_dist = -1
			max_cat = None

			v1_cat = new_data[i][1]
			v1 = new_data[i][0]

			for j in range(len(new_data)):
				if i == j:
					continue

				v2 = new_data[j][0]
				v2_cat = new_data[j][1]
				cos = self._get_cosine(v1, v2)

				if cos > max_dist:
					max_dist = cos
					max_cat = v2_cat

			if max_cat == v1_cat:
				success += 1
			else:
				error += 1

		return success, error

	def __init__(self, extracted_data):
		self.data = extracted_data

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

	def classify(self, doc):
		max_dist = -1
		max_cat = None

		v1 = Counter(doc[0] + doc[1])

		for j in range(len(self.train_data)):

			v2 = self.train_data[j][0]
			v2_cat = self.train_data[j][1]
			cos = self._get_cosine(v1, v2)

			if cos > max_dist:
				max_dist = cos
				max_cat = v2_cat

		return max_cat

	def __init__(self, train_data):
		new_train_data = []
		for doc in train_data:
			# [word_vec, category]
			new_train_data += [[Counter(doc[0] + doc[1]), doc[2]]]
		self.train_data = new_train_data

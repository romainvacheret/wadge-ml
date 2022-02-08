from typing import Tuple
from itertools import chain

from flask import Flask, request, jsonify

from kmeans import compute_kmeans # TODO change path

app = Flask('wadge-ml')


def transform_recipes(user: dict) -> Tuple[list[int], list[Tuple[int, int]]]:
	users_ids = []
	recipes_ids = []
	recipes_score = []

	from pprint import pprint
	# pprint(user)

	for recipe in user['recipes']:
		users_ids.append(user['id'])
		recipes_ids.append(recipe['recipe']['id'])
		recipes_score.append(recipe['score'])

	return users_ids, recipes_ids, recipes_score
	# user_id = user['id']
	# recipes = [(recipe['recipe']['id'], recipe['score']) \
	# 	for recipe in user['recipes']]

	# return [user_id] * len(recipes), recipes


def regroup(users: list[dict]):
	users_ids = []
	recipes_ids = []
	recipes_score = []

	for user in users:
		a, b, c = transform_recipes(user)
		users_ids += a 
		recipes_ids += b
		recipes_score += c

	return users_ids, recipes_ids, recipes_score

def get_user1_length(users: list[dict]) -> int:
	return [len(user['recipes']) for user in users if user['name'] == '1'][0]

def get_user1_id(users: list[dict]) -> int:
	return [user['id'] for user in users if user['name'] == '1'][0]

def to_tuple(user: dict) -> Tuple[int, int, int]:
	return [(user['id'], recipe['recipe']['id'], recipe['score']) \
		for recipe in user['recipes']]

def transform(users: list[dict]) -> list[Tuple[int, int, int]]:
	return list(chain(*map(to_tuple, users)))


def foo(tuples: list[Tuple[int, int, int]], 
	length: int, 
	id: int, clusters: list[int]):

	user1_recipes = [t for t in tuples if t[0] == id]
	user1_recipes_ids = [t[1] for t in user1_recipes] 
	top_10 = sorted(user1_recipes, key= lambda x: x[-1])[::-1][:10] # Must be sure there are at least 10
	print(top_10)
	top_10_idx = [tuples.index(t) for t in top_10]
	top_10_clusters = [clusters[idx] for idx in top_10_idx]
	print(top_10_clusters)
	return foo2(tuples, clusters, set(top_10_clusters)) - set(user1_recipes_ids)
	# top_10_ids = [t[1] for t in top_10]

def foo2(tuples, clusters, clusters_ids) -> list[int]:
	return set(t[1] for t, c  in zip(tuples, clusters) if c in clusters_ids)

class O:
	def __init__(self, u_id, r_id, score, label_=None) :
		self.u_id = u_id 
		self.r_id = r_id
		self.score = score
		self.label_ = label_

@app.route('/knn', methods=['POST'])
def compute_knn():
	if request.method == 'POST':
		body = request.json
		users = body['users']
		target = body['target']
		# transformed_recipes = [transform_recipes(user) for user in users]
		# y = list(zip(*(recipe for recipe in transformed_recipes))) 
		# print(y)
		# a, b = list(zip(*transformed_recipes))
		# # print(transformed_recipes)
		# print(b)
		# # TODO use KNN and return result
		# from pprint import pprint
		users_ids, recipes_ids, scores = regroup(users)
		# print(len(users_ids))
		# print(len(scores))
		x = compute_kmeans([[score] for score in scores])
		# print(x)
		# print(len(x))
		# print(transform(users))
		print(sum(len(user['recipes']) for user in users))
		user1_recipe_count = get_user1_length(users)
		user1_id = get_user1_id(users)
		result = foo(
			transform(users),
			user1_recipe_count,
			user1_id,
			x
		)

		print(result)
		# print(user1_id)

		return jsonify(list(result))


if __name__ == '__main__':
	app.run()
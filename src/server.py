from typing import Tuple
from flask import Flask, request, jsonify
from kmeans import compute_kmeans # TODO change path

app = Flask('wadge-ml')


def transform_recipes(user: dict) -> Tuple[list[int], list[Tuple[int, int]]]:
	users_ids = []
	recipes_ids = []
	recipes_score = []

	from pprint import pprint
	pprint(user)

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
		from pprint import pprint
		users_ids, recipes_ids, scores = regroup(users)
		print(len(users_ids))
		print(len(scores))
		x = compute_kmeans([[score] for score in scores])
		print(x)
		print(len(x))

		return jsonify([])


if __name__ == '__main__':
	app.run()
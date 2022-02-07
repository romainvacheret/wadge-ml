from typing import Tuple
from flask import Flask, request, jsonify


app = Flask('wadge-ml')


def transform_recipes(user: dict) -> list[Tuple[int, int, int]]:
	user_id = user['id']
	recipes = [(recipe['recipe']['id'], recipe['score']) \
		for recipe in user['recipes']]

	return [(user_id, *recipe) for recipe in recipes]


@app.route('/knn', methods=['POST'])
def compute_knn():
	if request.method == 'POST':
		body = request.json
		users = body['users']
		target = body['target']
		transformed_recipes = [transform_recipes(user) for user in users]
		print(transformed_recipes)

		# TODO use KNN and return result

		return jsonify([])


if __name__ == '__main__':
	app.run()
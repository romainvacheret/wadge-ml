from math import sqrt
from src.wadge_predict import ScoringRecipes

data_name = ["Person","Recipe", "Score"]
data = [(1, 1, 5), (1, 2, 1),(2, 1, 1), (2, 2, 5), (3, 1, 3), (3, 2, 3), (4, 1, 5), (4, 2, 1),(5, 1, 1), (5, 2, 5), (6, 1, 3), (6, 2, 3), (7, 1, 5), (7, 2, 1),(8, 1, 1), (8, 2, 5), (9, 1, 3), (9, 2, 3), (10, 1, 5), (10, 2, 1),(11, 1, 1), (11, 2, 5), (12, 1, 3), (12, 2, 3)]

# les labels associés aux enregistrements Clustering par notes -> positif = 0 // négatif = 1 // entre min et max/2 = 2 // entre max/2 et max = 3
# on recupere la note et on l'associe a un cluster 

le_max = -1000
le_min = 1000

for triplet in data:
    le_max = max(le_max, triplet[2])
    le_min = min(le_min, triplet[2])


target = []
for triplet in data:
    if triplet[2] == le_max :
        target.append(0)
    elif triplet[2] == le_min :
        target.append(1)
    elif triplet[2] > le_min and triplet[2] < sqrt(le_max):
        target.append(2)
    elif triplet[2] > sqrt(le_max) and triplet[2] < le_max :
        target.append(3)

model = ScoringRecipes(data=data, target=target, data_name=data_name)
model._make_knn(2)
model._res_knn()
model._print_score_matrice()
model._print_knn_diagram()

# TODO -> test main du DecisionTreeClassifier
# model_dtree = ScoringRecipes_dtree(data=data, target=target, data_name=data_name)
# model._make_dtree(2)
# model._res_dtree()
# model._print_score_matrice()
# model._print_dtree_diagram()
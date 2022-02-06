from math import sqrt
from src.wadge_knn import ScoringRecipesKnn
from src.wadge_dtree import ScoringRecipesDtree

data_name = ["Person","Recipe", "Score"] # Profil?
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

# test main du KNeighboursClassifier
model_knn = ScoringRecipesKnn(data=data, target=target)
model_knn.make_knn(2)
model_knn.res_knn()
model_knn.print_score_matrice()
model_knn.print_knn_diagram()

# test main du DecisionTreeClassifier
# model_dtree = ScoringRecipesDtree(data=data, target=target)
# model_dtree.make_dtree()
# model_dtree.res_dtree()
# model_dtree.print_score_matrice()
# model_dtree.print_dtree()
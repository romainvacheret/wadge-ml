from src.wadge_predict import ScoringRecipes

data_name = ["Person","Recipe", "Score"]
data = [(1, 1, 5), (1, 2, 1),(2, 1, 1), (2, 2, 5), (3, 1, 3), (3, 2, 3), (4, 1, 5), (4, 2, 1),(5, 1, 1), (5, 2, 5), (6, 1, 3), (6, 2, 3), (7, 1, 5), (7, 2, 1),(8, 1, 1), (8, 2, 5), (9, 1, 3), (9, 2, 3), (10, 1, 5), (10, 2, 1),(11, 1, 1), (11, 2, 5), (12, 1, 3), (12, 2, 3)]

# les labels associés aux enregistrements Clustering par notes -> 1 à 2 = 0 , 3 = 1, 4 à 5 = 2
#on recupere la note et on l'associe a un cluster 
target = []
for triplet in data:
    if(triplet[2] >= 4 ):
        target.append(2)
    elif(triplet[2] <= 2 ):
        target.append(0)
    elif(triplet[2] == 3):
        target.append(1)

model = ScoringRecipes(data=data, target=target, data_name=data_name)
model._make_ml_modele(2)
model._res_model()
model._print_score_matrice()
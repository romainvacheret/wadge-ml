from ctypes.wintypes import tagMSG
from sklearn import neighbors
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

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

print("Cluster", target)



data = pd.DataFrame(data)

data_train, data_test, target_train, target_test = train_test_split(data, target, random_state=0, train_size=0.5)


recipes_profil = neighbors.KNeighborsClassifier(n_neighbors=2)

recipes_profil.fit(data_train, target_train)

# résultat de la prédiction 
result_kn = recipes_profil.predict(data_test)
print("résultat : ", result_kn)

# matrice de confusion
conf_kn = confusion_matrix(result_kn, target_test)
# score sur la matrice
accuracy_kn = accuracy_score(result_kn, target_test)

print("accuracy : ", accuracy_kn)

sns.heatmap(conf_kn, square=True, annot=True, cbar=False)
plt.xlabel('valeurs prédites')
plt.ylabel('valeurs réelles')
plt.show()


# TODO -> récup les recettes K-means et ensuite prédire sur toutes les recettes en fonction de "? = ingrédients, notes, durée ..."

# k-means ou DecisionTree? lequel est le plus accurate
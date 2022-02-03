import numbers
from sklearn import neighbors
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd


class ScoringRecipes:
    def __init__(self, data: list, target: dict, data_name: list):
        self.data = data
        self.target = target
        self.data_name = data_name
        self.recipes_profil = []
        self.data_train = []
        self.data_test = []
        self.target_train = []
        self.target_test = []
        self.result = []
        self.matrice = []
        self.accuracy = 0
        self.faux_neg = 0
        self.faux_pos = 0
        


    def _make_ml_modele(self, nb_knn: int):
        data_frame = pd.DataFrame(self.data)
        self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(data_frame, self.target, random_state=0, train_size=0.5)

        # KNN Model -> k neighbors
        self.recipes_profil = neighbors.KNeighborsClassifier(n_neighbors= nb_knn)

        self.recipes_profil.fit(self.data_train, self.target_train)


    def _res_model(self):
        
        # résultat de la prédiction 
        self.result = self.recipes_profil.predict(self.data_test)

        # matrice de confusion
        self.matrice = confusion_matrix(self.result, self.target_test)

        # score sur la matrice
        self.accuracy = accuracy_score(self.result, self.target_test)
        self.faux_pos = precision_score(self.result, self.target_test, average="macro")
        self.faux_neg = recall_score(self.result, self.target_test, average="macro")

    def _print_score_matrice(self):

        print("Résultat : ", self.result)
        print("Accuracy : " + str(self.accuracy) + "\n" + "TPR : " + str(self.faux_pos) + "\n" + "FPR : " + str(self.faux_neg))

        sns.heatmap(self.matrice, square=True, annot=True, cbar=False)
        plt.xlabel('valeurs prédites')
        plt.ylabel('valeurs réelles')
        plt.show()


# TODO -> récup les recettes K-means et ensuite prédire sur toutes les recettes en fonction de "? = ingrédients, notes, durée ..."

# k-means ou DecisionTree? lequel est le plus accurate
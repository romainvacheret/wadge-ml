from sklearn import neighbors
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

class ScoringRecipesKnn:
    def __init__(self, data: list, target: dict):
        self.data = data
        self.target = target
        self.recipes_knn = []
        self.data_train = []
        self.data_test = []
        self.target_train = []
        self.target_test = []
        self.result_knn = []
        self.matrice = []
        self.accuracy = 0
        self.faux_neg = 0
        self.faux_pos = 0
        self.cross_knn = []

    def make_knn(self, nb_knn: int):
        self.data_frame = pd.DataFrame(self.data)
        self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(self.data_frame, self.target, random_state=0, train_size=0.5)

        # KNN Model -> k neighbors
        self.recipes_knn = neighbors.KNeighborsClassifier(n_neighbors= nb_knn)

        self.recipes_knn.fit(self.data_train, self.target_train)

    def res_knn(self):
        
        # résultat de la prédiction 
        self.result_knn = self.recipes_knn.predict(self.data_test)

        # cross validation
        self.cross_knn = cross_val_score(self.recipes_knn, self.data_frame, self.target )

        # matrice de confusion
        self.matrice = confusion_matrix(self.result_knn, self.target_test)

        # score sur la matrice
        self.accuracy = accuracy_score(self.result_knn, self.target_test)
        self.faux_pos = precision_score(self.result_knn, self.target_test, average="macro")
        self.faux_neg = recall_score(self.result_knn, self.target_test, average="macro")

    def print_score_matrice(self):

        print("Résultat KNN: ", self.result_knn)
        print("Cross Validation KNN: ", self.cross_knn)

        print("Accuracy KNN: " + str(self.accuracy) + "\n" + "TPR KNN: " + str(self.faux_pos) + "\n" + "FPR KNN: " + str(self.faux_neg))
        sns.heatmap(self.matrice, square=True, annot=True, cbar=False)
        plt.xlabel('valeurs prédites')
        plt.ylabel('valeurs réelles')
        plt.savefig("conf_matrice_knn.png")

    def print_knn_diagram(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        color = []
        for _ in self.target :
            if _ == 0 :
                color.append("#3DE70A") # vert = score très élevé
            if _ == 1 :
                color.append("#FF0000") # rouge = score très bas
            if _ == 2 :
                color.append("#FCFF00") # orange = score en dessous de la moyenne
            if _ == 3 :
                color.append("#B4C802") # jaune = score au dessus de la moyenne
        print(len(color))

        ax.scatter([triplet_score[2] for triplet_score in self.data], [triplet_id_person[0] for triplet_id_person in self.data], [triplet_recipe[1] for triplet_recipe in self.data], c=color)

        ax.set_xlabel('Score')
        ax.set_ylabel('Person')
        ax.set_zlabel('Recipes')
        plt.grid()
        plt.savefig("3d_dots_diagram_knn")

        # graphe des plus proches voisins
        # test = self.recipes_knn.kneighbors_graph(self.data_frame)
        # print(test.toarray())



# TODO -> train / allez chercher les "points autour" de l'utilisateur
# id_user = 1 -> trouver les recettes des users du même profil avec un score élevé (renvoyer les id_recipes)
# Ensuite prédire sur toutes les recettes en fonction de "? = ingrédients, notes, durée ..."

# TODO -> k-means ou DecisionTree? lequel est le plus accurate


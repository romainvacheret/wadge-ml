from enum import auto
from sklearn import tree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

class ScoringRecipesDtree :
    def __init__(self, data: list, target: dict):
        self.data = data
        self.target = target
        self.recipes_dtree = []
        self.data_train = []
        self.data_test = []
        self.target_train = []
        self.target_test = []
        self.result_dtree = []
        self.matrice = []
        self.accuracy = 0
        self.faux_neg = 0
        self.faux_pos = 0
        self.cross_dtree = []

    def make_dtree(self):
        self.data_frame = pd.DataFrame(self.data)
        self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(self.data_frame, self.target, random_state=0, train_size=0.5)

        self.recipes_dtree = tree.DecisionTreeClassifier()
        self.recipes_dtree.fit(self.data_train, self.target_train)

    # TODO -> erreur Axe3d + valeurs de prédictions fausses
    def res_dtree(self):

       # résultat de la prédiction 
        self.result_dtree = self.recipes_dtree.predict(self.data_test)

        # cross validation
        self.cross_dtree = cross_val_score(self.recipes_dtree, self.data_frame, self.target )

        # matrice de confusion
        self.matrice = confusion_matrix(self.result_dtree, self.target_test)

        # score sur la matrice
        self.accuracy = accuracy_score(self.result_dtree, self.target_test)
        self.faux_pos = precision_score(self.result_dtree, self.target_test, average="macro")
        self.faux_neg = recall_score(self.result_dtree, self.target_test, average="macro")

    def print_score_matrice(self):

        print("Résultat DTR: ", self.result_dtree)
        print("Cross Validation DTR: ", self.cross_dtree)

        print("Accuracy DTR: " + str(self.accuracy) + "\n" + "TPR DTR: " + str(self.faux_pos) + "\n" + "FPR DTR: " + str(self.faux_neg))
        sns.heatmap(self.matrice, square=True, annot=True, cbar=False)
        plt.xlabel('valeurs prédites')
        plt.ylabel('valeurs réelles')
        plt.savefig("conf_matrice_dtree.png")

    def print_dtree(self):
        plt.clf()
        tree.plot_tree(self.result_dtree, label="root")
        plt.show()
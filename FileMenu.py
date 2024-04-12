import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.insert(1, 'C://Users//user//OneDrive//fuculdade//OneDrive//Área de Trabalho//teste-algoritmo//DataMiningSamples-master//1-Preprocessing')
sys.path.insert(1, 'C://Users//user//OneDrive//fuculdade//OneDrive//Área de Trabalho//teste-algoritmo//DataMiningSamples-master//2-Clustering')
sys.path.insert(1, 'C://Users//user//OneDrive//fuculdade//OneDrive//Área de Trabalho//teste-algoritmo//DataMiningSamples-master//3-Classification')
import DataCleaning
import DataNormalization
import DataReduction
import GMM
import Kmeans
import DecisionTree
import DecisionTree_crossvalidation
import KNN
import KNN_crossvalidation
import SVM
import SVM_crossvalidation
import RedeNeural

def main():
    fileName = input("Selecione a base de dados que deseja realizar o pre-processamento de limpeza:\n1 - Mamo\n2 - Total Alcohol Consumption\n3 - Dummy\n") 
    outputFile = '0-Datasets/fileSend.data'

    if(fileName == "1"):
        names = ['BI-RADS','Age','Shape','Margin','Density','Severity']
        features = ['Age','Shape','Margin','Density']
        inputFile = '0-Datasets/Mamo.data'
        missingData = '?'
        target = 'Severity'
    if(fileName == "2"):
        names = ["Entity","Code","Year","Total alcohol consumption per capita (liters of pure alcohol, projected estimates, 15+ years of age)"]
        features = ["Entity","Code","Year"]
        inputFile = '0-Datasets/total-alcohol-consumption-per-capita-litres-of-pure-ahcool.csv'
        missingData = ',,'
        target = "Total alcohol consumption per capita (liters of pure alcohol, projected estimates, 15+ years of age)"
    if(fileName == "3"):
        names = ['age','gender','time_spent',"platform","interests","location","demographics",'profession','income','indebt','isHomeOwner','Owns_Car']
        features = ['age','gender','time_spent',"platform","interests","location","demographics",'profession','income','indebt','isHomeOwner']
        inputFile = '0-Datasets/dummy_data.csv'
        missingData = ""
        target = 'Owns_Car'
    if(fileName == "4"):
        names = ['pH', 'Temprature', 'Taste', 'Odor', 'Fat', 'Turbidity', 'Colour', 'Grade']
        features =  ['pH', 'Temprature', 'Taste', 'Odor', 'Fat', 'Turbidity', 'Colour']
        inputFile = '0-Datasets/milknew.csv'
        missingData = ""
        target = 'Grade'
    if(fileName == "5"):
        names = ['Class','age','menopause','tumor-size','inv-nodes','node-caps','deg-malig','breast','breast-quad','irradiat']
        features =  ['age','menopause','tumor-size','inv-nodes','node-caps','deg-malig','breast','breast-quad','irradiat']
        inputFile = '0-Datasets/breast-cancer.data'
        missingData = "?"
        target = 'Class'

    cleaningFeatures = features.copy()
    DataCleaning.dataCleaning(names, cleaningFeatures.append(target), inputFile, outputFile, missingData)

    selectMetode = input("Selecione o método de normalização da base de dados:\n1 - z-score\n2 - min-max\n")

    if(selectMetode == "1"):
        DataNormalization.dataNormalization(names, features, outputFile, target, 1)
    if(selectMetode == "2"):
        DataNormalization.dataNormalization(names, features, outputFile, target, 2)

    # Gerar um arquivo normalizado para que a etapa de redução possa utilizar
    # Criar um único nome de output para todas as saídas de pre-processamento, para que fique reescrevendo o mesmo arquivo economizando código

    selectMetode = input("Aperte '1' para realizar a reducao:")

    if(selectMetode == "1"):
        DataReduction.dataReduction(outputFile, names, features, target)

    selectMetode = input("Selecione o método de clustering da base de dados:\n1 - GMM\n2 - Kmeans\n")
    groupQnt = int(input("Digite a quantidade de grupos desejado:"))
    
    if(selectMetode == "1"):
        GMM.gmmFunction(outputFile, names, features, target, groupQnt)
    if(selectMetode == "2"):
        Kmeans.kmeansFunction(outputFile, names, features, target, groupQnt)
    
    selectMetode = input("Selecione o método de classificacao da base de dados:\n1 - Decision Tree\n2 - SVM\n3 - KNN\n4 - Rede Neural\n")
    if(selectMetode == "1"):
        qntFolhas = int(input("Digite a quantidade de folhas:"))
        DecisionTree.decisionTree(outputFile, names, features, target)
        DecisionTree_crossvalidation.decisionTreeCross(outputFile, names, features, target)
    if(selectMetode == "2"):
        SVM.svm(outputFile, names, features, target)
        SVM_crossvalidation.svmCross(outputFile, names, features, target)
    if(selectMetode == "3"):
        qntNN = int(input("Digite a quantidade de vizinhos:"))
        KNN.knn(outputFile, names, features, target, qntNN)
        KNN_crossvalidation.knnCross(outputFile, names, features, target, qntNN)
    if(selectMetode == "4"):
        qntHiddenLayer = int(input("Digite a quantidade de camadas:"))
        RedeNeural.redeNeural(outputFile, names, features, target, qntHiddenLayer)
    
if __name__ == "__main__":
    main()
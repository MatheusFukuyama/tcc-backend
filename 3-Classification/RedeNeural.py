from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler, SMOTE
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




def redeNeural(input_file, names, features, target, qntHiddenLayer):
    # names = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad',
    #          'irradiat']
    # features = ['age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast',
    #             'breast-quad', 'irradiat']
    # target = 'Class'
    # input_file = '0-Datasets/br-out.data'
    df = pd.read_csv(input_file,  # Nome do arquivo com dados
                     names=names)

    teste, target_names = pd.factorize(df.loc[:, target].values)

    # identify all categorical variables
    cat_columns = df.select_dtypes(['object']).columns

    # convert all categorical variables to numeric
    df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])

    # Separating out the features
    x = df.loc[:, features]
    # print(x, "aqui")

    # Separating out the target
    y = df.loc[:, target]

    smote = SMOTE(random_state = 32)
    x, y = smote.fit_resample(x, y) 
    
    # # Dividindo os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1)

    # Criar uma instância do MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(qntHiddenLayer, qntHiddenLayer), max_iter=2000, shuffle=False, random_state=1)

    # Treinar o modelo usando Holdout
    clf.fit(X_train, y_train)

    # Fazer previsões no conjunto de teste (Holdout)
    predictions_holdout = clf.predict(X_test)

   
    # Calcular acurácia e F1-score para Holdout
    accuracy_holdout = accuracy_score(y_test, predictions_holdout)
    f1_holdout = f1_score(y_test, predictions_holdout, average='macro')
    print("Holdout Metrics:")
    print("Accuracy: {:.2f}%".format(accuracy_holdout * 100))
    print("F1 Score: {:.2f}".format(f1_holdout * 100))

    cm = confusion_matrix(y_test, predictions_holdout)
    plot_confusion_matrix(cm, target_names, False, "Confusion Matrix - Rede Neural")
    plot_confusion_matrix(cm, target_names, True, "Confusion Matrix - Rede Neural normalized")  

    # Realizar Cross Validation
    scores_cv = cross_val_score(clf, x, y, cv=10)
    predictions_cv = cross_val_predict(clf, x, y, cv=10)

    # Calcular acurácia e F1-score para Cross Validation
    accuracy_cv = accuracy_score(y, predictions_cv)
    f1_cv = f1_score(y, predictions_cv, average='macro')
    print("Cross Validation Metrics:")
    print("Accuracy: {:.2f}%".format(accuracy_cv * 100))
    print("F1 Score: {:.2f}".format(f1_cv))

    plt.show()
if __name__ == "__main__":
    main()
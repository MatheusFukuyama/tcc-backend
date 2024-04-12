from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import f1_score
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
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


def decisionTree(input_file, names, features, target):
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
    x = df.loc[:, features].values

    # Separating out the target
    y = df.loc[:, [target]].values

    smote = SMOTE(random_state = 32)
    x, y = smote.fit_resample(x, y) 


    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    normalizedDf = pd.DataFrame(data=x, columns=features)
    print(normalizedDf)

    print(x)
    print(y)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)



    clf = DecisionTreeClassifier(max_leaf_nodes=45)
    print(X_train)
    clf.fit(X_train, y_train)
    tree.plot_tree(clf)
    plt.show()

    predictions = clf.predict(X_test)
    print(predictions)

    result = clf.score(X_test, y_test)
    f1 = f1_score(y_test, predictions, average='macro')
    print("Acurracy Decision Tree: {:.2f}%".format(result * 100))
    print("F1 Score Decision Tree: {:.2f}".format(f1))

    cm = confusion_matrix(y_test, predictions)
    plot_confusion_matrix(cm, target_names, False, "Confusion Matrix - K-NN sklearn")
    plot_confusion_matrix(cm, target_names, True, "Confusion Matrix - K-NN sklearn normalized" )
    plt.show()

if __name__ == "__main__":
    main()
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
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

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



def decisionTreeCross(input_file, names, features, target):
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

    clf = DecisionTreeClassifier(max_leaf_nodes=45)
    scores = cross_val_score(clf, x, y, cv=10, scoring= 'accuracy')
    predictions_cv = cross_val_predict(clf, x, y, cv=10)
    f1_cv = f1_score(y, predictions_cv, average='macro')

    media = np.array(scores)
    print(scores)
    print("accuracy: {:.2f}%" .format(media.mean()*100))
    print("F1 Score: {:.2f}%".format(f1_cv*100))

    cm = confusion_matrix(y, predictions_cv)
    plot_confusion_matrix(cm, target_names, False, "Confusion Matrix - K-NN sklearn")
    plot_confusion_matrix(cm, target_names, True, "Confusion Matrix - K-NN sklearn normalized" )
    plt.show()



if __name__ == "__main__":
    main()
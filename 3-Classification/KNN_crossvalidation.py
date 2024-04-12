# Initial imports
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict


# Calculate distance between two points
def minkowski_distance(a, b, p=1):
    # Store the number of dimensions
    dim = len(a)
    # Set initial distance to 0
    distance = 0

    # Calculate minkowski distance using parameter p
    for d in range(dim):
        distance += abs(a[d] - b[d])**p

    distance = distance**(1/p)
    return distance


def knn_predict(X_train, X_test, y_train, y_test, k, p):
    # Make predictions on the test data
    # Need output of 1 prediction per test data point
    y_hat_test = []

    for test_point in X_test:
        distances = []

        for train_point in X_train:
            distance = minkowski_distance(test_point, train_point, p=p)
            distances.append(distance)

        # Store distances in a dataframe
        df_dists = pd.DataFrame(data=distances, columns=['dist'],
                                index=y_train.index)

        # Sort distances, and only consider the k closest points
        df_nn = df_dists.sort_values(by=['dist'], axis=0)[:k]

        # Create counter object to track the labels of k closest neighbors
        counter = Counter(y_train[df_nn.index])

        # Get most common label of all the nearest neighbors
        prediction = counter.most_common()[0][0]

        # Append prediction to output list
        y_hat_test.append(prediction)

    return y_hat_test


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


def knnCross(input_file, names, features, target, qntNN):
    # Load iris data and store in dataframe
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
   
    print("Total samples: {}".format(x.shape[0]))

    # Separating out the target
    y = df.loc[:, target]
 
    smote = SMOTE(random_state = 32)
    x, y = smote.fit_resample(x, y)

    x = StandardScaler().fit_transform(x)

    mean_max = 0
    indice_max = 0
    f1_score_max = 0
    for i  in range(100):
        i = i + 1

        knn = KNeighborsClassifier(n_neighbors=i)

        scores = cross_val_score(knn, x, y, cv=10, scoring= 'accuracy')
        predictions_cv = cross_val_predict(knn, x, y, cv=10)
        f1_cv = f1_score(y, predictions_cv, average='macro')

        media = np.array(scores).mean()

        if  media > mean_max:
            mean_max = media
            indice_max = i
            f1_score_max = f1_cv
    
    print(i)
    print(scores)
    print("accuracy: {:.2f}%" .format(mean_max*100))
    print("F1 Score: {:.2f}%".format(f1_score_max*100))



if __name__ == "__main__":
    main()
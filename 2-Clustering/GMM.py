#Implementation of Kmeans from scratch and using sklearn
#Loading the required modules 
import numpy as np
from scipy.spatial.distance import cdist 
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import homogeneity_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

def plot_samples(projected, labels, title):    
    fig = plt.figure()
    u_labels = np.unique(labels)
    for i in u_labels:
        plt.scatter(projected[labels == i , 0] , projected[labels == i , 1] , label = i,
                    edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('tab10', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.legend()
    plt.title(title)

def gmmFunction(input_file, names, features, target, groupQnt):
    # Load dataset Digits
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
    y = df.loc[:, target].values

    x = StandardScaler().fit_transform(x)
    normalizedDf = pd.DataFrame(data=x, columns=features)
    normalizedDf = pd.concat([normalizedDf, df[target]], axis=1)
    
    #Transform the data using PCA
    pca = PCA(2)
    projected = pca.fit_transform(normalizedDf)
    
    #Applying sklearn GMM function
    gm  = GaussianMixture(n_components=groupQnt).fit(projected)
    print(gm.weights_)
    print(gm.means_)
    x = gm.predict(projected)

    print(x)
    print(projected)

    score = silhouette_score(projected, x)    
    print("For n_clusters = {}, silhouette score is {})".format(4, score))
    print(homogeneity_score(x, y))

    #Visualize the results sklearn
    plot_samples(projected, x, 'Clusters Labels GMM')

    plt.show()

if __name__ == "__main__":
    main()
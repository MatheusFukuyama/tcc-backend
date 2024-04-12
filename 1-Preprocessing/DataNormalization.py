import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def dataNormalization(names, features, input, target, normMetodo):
    # Faz a leitura do arquivo
    # input_file = '../0-Datasets/MamoClear.data'
    # names = ['Age','Shape','Margin','Density','Severity']
    # features = ['Age','Shape','Margin','Density']
    # target = 'Severity'
    df = pd.read_csv(input,    # Nome do arquivo com dados
                     names = names) # Nome das colunas                      
    ShowInformationDataFrame(df,"Dataframe original")

    #identify all categorical variables
    cat_columns = df.select_dtypes(['object']).columns
    print(cat_columns)
    # ShowInformationDataFrame(a,"Dataframe ordenado")

    # convert all categorical variables to numeric
    df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])
    # df.to_csv(output_file, header=False, index=False)


    # Separating out the features
    x = df.loc[:, features].values
    
    # Separating out the target
    y = df.loc[:,[target]].values

    if(normMetodo == 1):
        # Z-score normalization
        x_zcore = StandardScaler().fit_transform(x)
        normalized1Df = pd.DataFrame(data = x_zcore, columns = features)
        normalized1Df = pd.concat([normalized1Df, df[[target]]], axis = 1)
        ShowInformationDataFrame(normalized1Df,"Dataframe Z-Score Normalized")

    if(normMetodo == 2):
        # Mix-Max normalization
        x_minmax = MinMaxScaler().fit_transform(x)
        normalized2Df = pd.DataFrame(data = x_minmax, columns = features)
        normalized2Df = pd.concat([normalized2Df, df[[target]]], axis = 1)
        ShowInformationDataFrame(normalized2Df,"Dataframe Min-Max Normalized")


def ShowInformationDataFrame(df, message=""):
    print(message+"\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n") 


if __name__ == "__main__":
    main()
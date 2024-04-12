import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def dataDispartion(input_file, names, features):
    # names = ['Class','age','menopause','tumor-size','inv-nodes','node-caps','deg-malig','breast','breast-quad','irradiat'] 
    # features =  ['Class','age','menopause','tumor-size','inv-nodes','node-caps','deg-malig','breast','breast-quad','irradiat'] 
    df = pd.read_csv('0-DataSets/br-out.data', names = names)

    print(df.head(15))

    df_dispersao = df[['age', 'Class']]
    df_dispersao = df_dispersao.groupby(['age', 'Class']).size().reset_index(name="tamanho")
    fig = plt.figure()
    ax = fig.add_axes([0,0,1.5,1.5])

    ax = sns.scatterplot(data=df_dispersao, x='Class', y = "age", legend=False, sizes=(1000,10000))

    plt.show()

if __name__ == "__main__":
    main()


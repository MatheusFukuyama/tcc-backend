import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def dataCorrelation(input_file, names, features):
    # names = ['Class','age','menopause','tumor-size','inv-nodes','node-caps','deg-malig','breast','breast-quad','irradiat'] 
    # features =  ['Class','age','menopause','tumor-size','inv-nodes','node-caps','deg-malig','breast','breast-quad','irradiat'] 
    df = pd.read_csv('../0-DataSets/normalizedFile.data', names = names)

    correlation = df.corr()
    plot = sn.heatmap(correlation, annot = True, fmt=".1f", linewidths=.6)
    
    plt.show()
if __name__ == "__main__":
    main()
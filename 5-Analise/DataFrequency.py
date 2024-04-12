import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ggplot as ggplot

def dataFrequency(input_file, names, features):
    # names = ['Class','age','menopause','tumor-size','inv-nodes','node-caps','deg-malig','breast','breast-quad','irradiat'] 
    # features =  ['Class','age','menopause','tumor-size','inv-nodes','node-caps','deg-malig','breast','breast-quad','irradiat'] 
    df = pd.read_csv('../0-DataSets/normalizedFile.data', names = names)

    print(df.head(15))


    df['inv-nodes'].hist(by=df['node-caps'], bins=20)
    df['age'].hist(by=df['menopause'])
    p1 = ggplot(data, aes(x=inv.nodes, fill=Class)) + geom_bar(position='dodge') + labs(title='Histogram of Inv Nodes Grouped by Class',x='Inv Nodes',y='Count')

    df['age'].hist(bins=20)

    
    
     
    plt.show()

if __name__ == "__main__":
    main()
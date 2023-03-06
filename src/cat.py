import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


def catClass(df_n):

    hate2 = df_n
    hate_list = hate2['id'].values.tolist()

    categoryDict = {'Bad':[],'Worst':[],'Extreme':[]}

    count_list = []
    for i in hate_list:
        count_list.append(hate_list.count(i))

    for j in range(0,989):
        if count_list[j]==3:
            categoryDict['Bad'].append(hate_list[j])
            categoryDict['Bad'] = list(dict.fromkeys(categoryDict['Bad']))
        elif count_list[j]==4:
            categoryDict['Worst'].append(hate_list[j])
            categoryDict['Worst'] = list(dict.fromkeys(categoryDict['Worst']))
        elif count_list[j]==5:
            categoryDict['Extreme'].append(hate_list[j])
            categoryDict['Extreme'] = list(dict.fromkeys(categoryDict['Extreme']))

    return categoryDict

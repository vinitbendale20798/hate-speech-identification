import pandas as pd
import numpy as np

def bannedTweets(hate,ban):
    
    hate2 = hate
    ban_list = ban
    
    arr1 = []
    arr2 = []
    arr3 = []

    for i in range(0,len(ban_list)):
        for j in range(0,989):
            if(hate2.iloc[j]['id'] == ban_list[i]):
                arr1.append(hate2.iloc[j]['id'])
                arr2.append(hate2.iloc[j]['label'])
                arr3.append(hate2.iloc[j]['Tweet'])
         
    banned_df = pd.DataFrame({'id':arr1,'label':arr2,'Tweet':arr3})

    banned_df.to_csv('D:/Major/outputs/banned_tweets.csv',index=False)

    return 0

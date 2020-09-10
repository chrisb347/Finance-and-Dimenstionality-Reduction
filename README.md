# Finance and Dimensionality Reduction
 Examples of models leveraging financial data

The slides in this directory have been put together well, and the code on how we achieved each result is in the colab doc. For this project, we were given 8-9 data sets so it was advantageous to write many functions for this project and stick them into loops. 

## Finding 1
Our first finding was best displayed on a triple bar graph, where we could easily show the phenomena we observed. The question asked us to display 3 different data transformations on a data set and all of the PCA's for each. we used the **Raw Data**, **Min Max Scaler**, and the **Standard Scaler** methods for transforming the data. The graph below is interesting because it shows how the variance is attributed to the first PCA when there are more variables in the data set in the **Raw Data**. With a dataset with less variables, the first PCA magnitude is more consistend between the **Raw Data** and **Transformed Data**

![Alt text](/images/PCA_EVR.PNG?raw=true "Optional Title")

```
#data=hft_aapl
#import matplotlib.pyplot as plt
# high price and low price embedding
price_hl_embedding = data['marketHigh']
x=pd.concat([price_hl_embedding,pd.get_dummies(db_outlier_label)[1]],axis=1)

x.index=hft_aapl['Date']
x['outlier']=x['marketHigh']*x.iloc[:,1]
marks=x[1]

dates=x[1]==1

dates
x1=x[dates]
dates=x1.index
x=x['marketHigh']


def find_loc(df, dates):
    marks = []
    for date in dates:
        marks.append(df.index.get_loc(date))
    return marks

plt.figure(figsize=(20,10))
x.plot(linestyle='-',markevery=find_loc(x,dates), marker='o',markerfacecolor='black')
#plt.show()
#x['label'] = 1 if x[0]=='True' else 0

plt.show()
```






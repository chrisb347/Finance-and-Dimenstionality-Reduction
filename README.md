# Finance and Dimensionality Reduction
 Examples of models leveraging financial data

The slides in this directory have been put together well, and the code on how we achieved each result is in the colab doc. For this project, we were given 8-9 data sets so it was advantageous to write many functions for this project and stick them into loops. 

## Finding 1
Our first finding was best displayed on a triple bar graph, where we could easily show the phenomena we observed. The question asked us to display 3 different data transformations on a data set and all of the PCA's for each. we used the **Raw Data**, **Min Max Scaler**, and the **Standard Scaler** methods for transforming the data. The graph below is interesting because it shows how the variance is attributed to the first PCA when there are more variables in the data set in the **Raw Data**. With a dataset with less variables, the first PCA magnitude is more consistend between the **Raw Data** and **Transformed Data**

```

##transformation functions
  def doPCA_std (data):
    scalar = StandardScaler()
    normalized_data=scalar.fit_transform(data)
    pca=PCA()
    newPCAData=pca.fit_transform(normalized_data)
    return(newPCAData,pca)
  
  def doPCA (data):
    pca=PCA()
    newPCAData=pca.fit_transform(data)
    return(newPCAData,pca)
  
  def doPCA_minMax (data):
    minMax_data=MinMaxScaler().fit_transform(data)
    minMax_data_df=pd.DataFrame(data=minMax_data, index=data.index,
                            columns=data.columns)
    pca=PCA()
    newPCAData=pca.fit_transform(minMax_data_df)
    return(newPCAData,pca)
    
##loop through data sets and put them into dictionaries     
  newData=dict()
  raw_pca=dict()
  newData=dict()
  std_pca=dict()
  newData=dict()
  minMax_pca=dict()


  for i in data_names1:
    newData[i],raw_pca[i]=doPCA(data_names1[i])
  
  for i in data_names1:
     newData[i],std_pca[i]=doPCA_std(data_names1[i])
   
   for i in data_names1:
     newData[i],minMax_pca[i]=doPCA_minMax(data_names1[i])
     
#put the explained variance ratios on a graph     
for i in minMax_pca:
# set width of bar
  barWidth = 0.25
 
# set height of bar
  bars1 = minMax_pca[i].explained_variance_ratio_ * 100
  bars2 = std_pca[i].explained_variance_ratio_ * 100
  bars3 = raw_pca[i].explained_variance_ratio_ * 100
 
# Set position of bar on X axis
  r1 = np.arange(len(bars1))
  r2 = [x + barWidth for x in r1]
  r3 = [x + barWidth for x in r2]
 
# Make the plot
  fig=plt.figure(figsize=(20,7))
  plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Min Max')
  plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='Standard Scaler')
  plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='Raw Data')
 
# Add xticks on the middle of the group bars
  plt.xlabel('group', fontweight='bold')
  plt.xticks([r + barWidth for r in range(len(bars1))], ['PC' + str(q) for q in range(1, len(bars1) + 1)])
  plt.title(i+" "+'Explained Varinace Ratios')
  plt.ylabel('percentange of explained variance')
  plt.xlabel('principal component')
# Create legend & Show graphic
  plt.legend()

  plt.show()
  
```

![Alt text](/images/PCA_EVR.PNG?raw=true "Optional Title")


Next I mapped the db scan outlier labels to the high frequency trading pricing data. It can be observed below that db scan does a good job of finding localized outliers. This can be useful in finding buy and sell points in a high frequency trading framework.  

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

![Alt text](/imagesOutlier_price_space.PNG?raw=true "Optional Title")







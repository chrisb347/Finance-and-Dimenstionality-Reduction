# Finance and Dimensionality Reduction
 Examples of models leveraging financial data

The slides in this directory have been put together well, and the code on how I achieved each result is in the colab doc. For this project, we were given 8-9 data sets so it was advantageous to write many functions for this project and stick them into loops. Below are some samples of code and findings from this project 

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

## Finding 2

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

![Alt text](/images/Outliers_price_space.png?raw=true "Optional Title")

## Finding 3

A part of the project was to find a high frequency data provider on our own, download the data ata second level and provide analysis on the data. I went out and found Polygon.io, which is a high frequency provider that can provide second level stock ticker data. Their API has a module of python functions that the user can fill out to connect to the API, and download *.json* files for as many tickers you pass into the API.

I compared the two TSNE components, and labeled the stocks by direction of the open from the close. If it went up or down it was labeled blue or green respectively. If it was equal then it was labeled yellow, which is the interesting part of this analysis. The yellow points for every single stock seemed to cluster together. I argued this may signify a change in price direction and TSNE is able to identify this. This could be helpful in creating a high frequency trading strategy.

```
# Get t-SNE Embedding
def do_TSNE(data, perplexity=50, init='pca'):
    normalized_data = StandardScaler().fit_transform(data)
    tsne_ = TSNE(perplexity=perplexity,
                 init=init)
    tsneNewData = tsne_.fit_transform(normalized_data)
    return tsneNewData

##biplot function
def biplot(embedding, label, label_name, Method, size=(10,7), outlier_label=None):
    cmap = plt.get_cmap("tab20")
    mk_list = ["o","p",">","X","P"]
    fig, ax = plt.subplots(figsize=size)
    plt.style.use('seaborn')
    for i in range(len(label)):
        ax.scatter(embedding[label[i],0], embedding[label[i],1], label=label_name[i],
                   marker=mk_list[i], color=cmap(2*i), edgecolor='w', alpha=0.7, s=75)
    if outlier_label is not None:
        ax.scatter(embedding[outlier_label,0], embedding[outlier_label,1], 
                   label='Outlier', marker='s', color='black', edgecolor='gold', alpha=0.5, s=60)
    ax.legend(prop={'size': 10})
    x_text = '$PC_{1}$' if Method == 'pca' else ('$SparsePC_{1}$' if Method == 'spca' else ('$'+Method+'_{1}'+'$'))
    y_text = '$PC_{2}$' if Method == 'pca' else ('$SparsePC_{2}$' if Method == 'spca' else ('$'+Method+'_{2}'+'$'))
    ax.set_xlabel(x_text, fontsize=14)
    ax.set_ylabel(y_text, fontsize=14)
    plt.show()
    
    return fig, ax


###execute TSNE on High Frequemcy Trading data

for i in ticker_list:
  col_rename = {'v':'Volume', 'vw':'Volume Weighted Average Price', 'o':'Open', 'c':'Close', 'h':'High', 'l':'Low', 't':'Timestamp', 'n':'NumberOfItems'}

  hft_dict[i].rename(columns=col_rename, inplace=True)

#hft_dict['AMZN'] = hft_dict['AMZN'][hft_dict['AMZN'].applymap(np.isreal).any(1)]

  hft_dict[i]=hft_dict[i].dropna()

  
  tsne_embedding = do_TSNE(hft_dict[i], perplexity=50, init='pca')
  hft_dict[i]['Change'] = hft_dict[i]['Close'] - hft_dict[i]['Open']
  hft_dict[i]['Change'] = hft_dict[i]['Change'].apply(lambda x: 'Rise' if x>0 else ('Fall' if x<0 else 'Flat'))
  change_ = hft_dict[i]['Change']
  oc_label = [hft_dict[i]['Change'] == 'Rise', hft_dict[i]['Change'] == 'Flat', hft_dict[i]['Change'] == 'Fall']
  oc_label_name = ['Close > Open', 'Close = Open', 'Close < Open']
  #save_pickle(tsne_embedding, 'tsne_embedding')
  fig, ax = biplot(tsne_embedding, oc_label, oc_label_name, 'tsne', size=(10,7))
  print(i)    

```  

![Alt text](/images/TSNE_on_HFT.PNG?raw=true "Optional Title")  








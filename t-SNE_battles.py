import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# step 1: download the data and give it's path
#Go to the Kaggle page and download the dataset (https://www.kaggle.com/mylesoneill/game-of-thrones)
dataframe_all = pd.read_csv("./game-of-thrones/battles.csv")
num_rows = dataframe_all.shape[0]

# step 2: remove useless data
# count the number of missing elements (NaN) in each column
counter_nan = dataframe_all.isnull().sum()
counter_without_nan = counter_nan[counter_nan==0]

#remove the columns with missing elements
dataframe_all = dataframe_all[counter_without_nan.keys()]

all_val=dataframe_all

# remove unnecessary data
dataframe_all = dataframe_all.drop(dataframe_all.columns[[0,3,4]],axis=1)

# step 3: get features (x) and scale the features
# get x and convert it to numpy array
x = dataframe_all.ix[:,:-1].values
standard_scaler = StandardScaler()
x_std = standard_scaler.fit_transform(x)


# step 4: get class labels y and then encode it into number 
# get class label data
y = all_val['region']
# encode the class label
class_labels = np.unique(y)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# t-distributed Stochastic Neighbor Embedding (t-SNE) visualization
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
x_std_2d = tsne.fit_transform(x_std)

#define markers and colors depending on the number of classes (in this case 7 classes equal to battle regions in GoT)
markers=(',','o','v','^','<','>','1')
color_map = {0:'red', 1:'blue', 2:'lightgreen', 3:'purple', 4:'cyan', 5:'yellow', 6:'orange'}

#plot the data
plt.figure()
for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=x_std_2d[y==cl,0], y=x_std_2d[y==cl,1], c=color_map[idx], marker=markers[idx], label=cl)
plt.xlabel('X in t-SNE')
plt.ylabel('Y in t-SNE')
plt.legend(loc='upper left')
plt.title('t-SNE visualization of test data')
plt.show()
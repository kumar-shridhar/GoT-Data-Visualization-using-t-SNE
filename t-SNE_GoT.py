import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# step 1: download the data and give it's path
#Go to the Kaggle page and download the dataset (https://www.kaggle.com/mylesoneill/game-of-thrones)
dataframe_all = pd.read_csv("./game-of-thrones/character-deaths.csv")
num_rows = dataframe_all.shape[0]

# step 2: remove useless data
# count the number of missing elements (NaN) in each column
counter_nan = dataframe_all.isnull().sum()
counter_without_nan = counter_nan[counter_nan==0]
#remove the columns with missing elements
dataframe_all = dataframe_all[counter_without_nan.keys()]
#all_val variable defined to use later for labels
all_val=dataframe_all
# remove the first 2 columns 'Name' and 'Allegiances'
dataframe_all = dataframe_all.ix[:,2:]
columns = dataframe_all.columns

# step 3: get features (x) and scale the features
# get x and convert it to numpy array
x = dataframe_all.ix[:,:-1].values
standard_scaler = StandardScaler()
x_std = standard_scaler.fit_transform(x)

# step 4: get class labels y and then encode it into number 
# get class label data
y = all_val['Allegiances']
# encode the class label
class_labels = np.unique(y)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# step 5: split the data into training set and test set
test_percentage = 0.1
x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size = test_percentage, random_state = 0)

# t-distributed Stochastic Neighbor Embedding (t-SNE) visualization
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
x_test_2d = tsne.fit_transform(x_test)

#define markers and colors depending on the number of classes (in this case 21 classes equal to houses in GoT)
markers=(',','o','v','^','<','>','1','2','3','4','s','p','*','h','H','+','x','D','d','|','_')
color_map = {0:'red', 1:'blue', 2:'lightgreen', 3:'purple', 4:'cyan', 5:'yellow', 6:'orange', 7:'green', 8:'black', 
            9:'blue',10:'blue',11:'blue',12:'blue',13:'blue',14:'blue',15:'blue',16:'blue',17:'blue',18:'blue',19:'blue',20:'blue',
            21:'blue'}

#plot the data
plt.figure()
for idx, cl in enumerate(np.unique(y_test)):
    plt.scatter(x=x_test_2d[y_test==cl,0], y=x_test_2d[y_test==cl,1], c=color_map[idx], marker=markers[idx], label=cl)
plt.xlabel('X in t-SNE')
plt.ylabel('Y in t-SNE')
plt.legend(loc='upper left')
plt.title('t-SNE visualization of test data')
plt.show()
# GoT-Data-Visualization-using-t-SNE
Visualize multidimensional Game of Thrones data-set using t-SNE.

1. Dataset downloaded from Kaggle: https://www.kaggle.com/mylesoneill/game-of-thrones

2. Run the python file:

    t-SNE_GoT.py



3.Print the label meaning by printing the class_labels as:
 	
 	print class_labels

In this case the 21 labels are:

 ['Arryn' 'Baratheon' 'Greyjoy' 'House Arryn' 'House Baratheon'
 'House Greyjoy' 'House Lannister' 'House Martell' 'House Stark'
 'House Targaryen' 'House Tully' 'House Tyrell' 'Lannister' 'Martell'
 "Night's Watch" 'None' 'Stark' 'Targaryen' 'Tully' 'Tyrell' 'Wildling']


 We can visualize how members of one House is close to members of other Houses based on the how close they lie to each other. We can visualize in the left big cluster that some 'Martells' are close to each other while one 'Martell' is closer to 'Stark'. Also, some 'Night's Watch' are grouped together. 
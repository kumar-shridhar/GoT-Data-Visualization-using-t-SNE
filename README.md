# GoT-Data-Visualization-using-t-SNE
Visualize multidimensional Game of Thrones data-set using t-SNE.


DATASET

Dataset downloaded from Kaggle: https://www.kaggle.com/mylesoneill/game-of-thrones


BATTLES


1.Run the python file:

	t-SNE_battles.py


2.Class labels used here are:

['Beyond the Wall' 'The Crownlands' 'The North' 'The Reach'
 'The Riverlands' 'The Stormlands' 'The Westerlands']


We can visualize the data as in which year battles in what region took place. The regions are used as labels here.




CHARACTER DEATH


1.Run the python file:

	t-SNE_charDeath.py



2.Print the label meaning by printing the class_labels as:
 	
 	print class_labels

In this case the 21 labels are:

 ['Arryn' 'Baratheon' 'Greyjoy' 'House Arryn' 'House Baratheon'
 'House Greyjoy' 'House Lannister' 'House Martell' 'House Stark'
 'House Targaryen' 'House Tully' 'House Tyrell' 'Lannister' 'Martell'
 "Night's Watch" 'None' 'Stark' 'Targaryen' 'Tully' 'Tyrell' 'Wildling']


 We can visualize how members of one House is close to members of other Houses based on the how close they lie to each other. We can visualize in the left big cluster that some 'Martells' are close to each other while one 'Martell' is closer to 'Stark'. Also, some 'Night's Watch' are grouped together. 





 CHARACTER PREDICTION


1.Run the python file:

 	t-SNE_charPred.py


2.The characters are visualized as whether alive or dead with 0 as dead (in Red) and 1 as alive (in Blue).


We can visualize some dead and alive characters together. We cannot say for sure that the alive characters near to dead characters will be dead soon, but we can say for sure that those alive characters have similarity with dead characters like they might be popular and noble. E.g: Robb Stark and Sansa Stark.

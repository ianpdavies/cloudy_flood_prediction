import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import GridSearchCV

# CSV of sample points with features
points = pd.read_csv('../Data/sample_points2.csv')

# points.head()
list(points.columns)

# Recode landcover categories as dummy variables
# Probably want to remove impervious as well, or collapse some of the landcover types
nlcd_dict = {11:'open_water', # Dict of landcover values
             12:'perm_snowIce',
             21:'dev_openSpace',
             22:'dev_lowInt',
             23:'dev_medInt',
             24:'dev_highInt',
             31:'barren',
             41:'decid_forest',
             42:'evergreen_forest',
             43:'mixed_forest',
             51:'dwarf_scrub',
             52:'shrub',
             71:'grassland',
             72:'sedge',
             73:'lichens',
             74:'moss',
             81:'pasture',
             82:'crops',
             90:'woody_wetlands',
             95:'emergent_wetlands'}

df = points.replace({'landcover':nlcd_dict})

dummies = pd.get_dummies(df['landcover']).rename(columns=lambda x: str(x)) # Convert categorical feature to dummy vars
df = pd.concat([df, dummies], axis=1) # Concat with other features
df = df.drop(['landcover'], axis=1) # Drop categorical landcover feature
df = df.reset_index()

# Rescale values
from sklearn.preprocessing import MinMaxScaler
# imageID = df.pop('imageID') # Pop out imageID
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
# df_scaled = pd.concate(df_scaled, imageID) # Add imageID back 

print(df_scaled)
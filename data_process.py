import pandas as pd
import numpy as np

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
import seaborn as sns

%matplotlib inline


data = pd.read_csv('filename.csv')
data.describe()
data.info()

#divide into test and training
train, test = train_test_split(data, test_size = 0.2)

#vitualising

#setting color palette
red_blue = ["#19B5FE", "#EF4836"]
palette = sns.color_palette(red_blue)
sns.set_palette(palette)
sns.set_style("white")

#setting variables for plotting
skip1_pos_tempo = data[data['target'] == 1]['tempo']
skip1_neg_tempo = data[data['target'] == 0]['tempo']
skip1_pos_dance = data[data['target'] == 1]['danceability']
skip1_neg_dance = data[data['target'] == 0]['danceability']
skip1_pos_duration = data[data['target'] == 1]['duration_ms']
skip1_neg_duration = data[data['target'] == 0]['duration_ms']
skip1_pos_loudness = data[data['target'] == 1]['loudness']
skip1_neg_loudness = data[data['target'] == 0]['loudness']


#plot1
fig1 = plt.figure(figsize = (15,15))

#danceability
ax3 = fig1.add_subplot(331)
ax3.set_xlabel('Danceability')
ax3.set_ylabel('Count')
ax3.set_title('Song Danceability Skip Distribution')
skip1_pos_dance.hist(alpha = 0.5, bins = 30)
ax4 = fig1.add_subplot(331)
skip1_neg_dance.hist(alpha = 0.5, bins = 30)

#beat_strength
ax5 = fig1.add_subplot(332)
ax5.set_xlabel('Duration(ms)')
ax5.set_ylabel('Count')
ax5.set_title('Song Duration Skip Distribution')
skip1_pos_duration.hist(alpha = 0.5, bins = 30)
ax6 = fig1.add_subplot(332)
skip1_neg_duration.hist(alpha = 0.5, bins = 30)

#dyn_range_mean
ax7 = fig1.add_subplot(333)
skip1_pos_loudness.hist(alpha = 0.5, bins = 30)
ax7.set_xlabel('Loudness')
ax7.set_ylabel('Count')
ax7.set_title('Song Loudness Skip Distribution')

ax8 = fig1.add_subplot(333)
skip1_neg_loudness.hist(alpha = 0.5, bins = 30)

#organism 
ax9 = fig1.add_subplot(334)
skip1_pos_speechiness.hist(alpha = 0.5, bins = 30)
ax9.set_xlabel('speechiness')
ax9.set_ylabel('Count')
ax9.set_title('Song Speechiness Skip Distribution')

ax10 = fig1.add_subplot(334)
skip1_neg_speechiness.hist(alpha = 0.5, bins = 30)

#Acousticness
ax17 = fig1.add_subplot(335)
skip1_pos_acousticness.hist(alpha = 0.5, bins = 30)
ax17.set_xlabel('acousticness')
ax17.set_ylabel('Count')
ax17.set_title('Song acousticness Skip Distribution')

ax18 = fig1.add_subplot(335)
skip1_neg_acousticness.hist(alpha = 0.5, bins = 30)

#mechanism
ax19 = fig1.add_subplot(336)
skip1_pos_instrumentalness.hist(alpha = 0.5, bins = 30)
ax19.set_xlabel('instrumentalness')
ax19.set_ylabel('Count')
ax19.set_title('Song Instrumentalness Skip Distribution')

ax20 = fig1.add_subplot(336)
skip1_neg_instrumentalness.hist(alpha = 0.5, bins = 30)

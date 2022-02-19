import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import pickle
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation



header = st.container()
dataset = st.container()
fearutrs = st.container()
model_training = st.container()


with header:
	st.title('Recommendation Engine for a Devotional Songs!') 
	st.text('In this project I look into the popularity of songs.')


with dataset:
	st.header('Millian song dataset')
	
	song_dict1 = pickle.load(open('data/df2song.pkl','rb'))
	df = pd.DataFrame(song_dict1)
	song_dict = pickle.load(open('data/msong.pkl','rb'))
	song = pd.DataFrame(song_dict)
	st.write(song.head(20))

	users = song.pivot_table(index ='song_id',columns = 'artist_name', values = 'title').reset_index(drop = True)
	users.index=song['song_id'].unique()
	users.fillna(0,inplace = True)
	user_sim = 1 - pairwise_distances(users.values,metric = 'cosine')
	user_sim_df = pd.DataFrame(user_sim)
	user_sim_df = pd.DataFrame(user_sim)
	user2 = user_sim_df.iloc[0:10,0:10]
	np.fill_diagonal(user_sim,0)
	user2 = user_sim_df.iloc[0:10,0:10]
	user_sim_df.idxmax(axis = 1)[0:50]
	selected_song_id1 = st.selectbox('Song_Id_1',user_sim_df.idxmax(axis = 1)[0:50])
	
	
	if st.button('Recommend'):
		
		st.write (df[(song['song_id']==selected_song_id1)])

	
	

		
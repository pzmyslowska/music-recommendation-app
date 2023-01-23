# In[25]:


from tkinter import *

import numpy as np
import pandas as pd

import plotly.express as px 

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings("ignore")

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
from scipy.spatial.distance import cdist


# In[26]:


data = pd.read_csv("data.csv")
genre_data = pd.read_csv('data_by_genres.csv')
year_data = pd.read_csv('data_by_year.csv')


# In[27]:
data.head()
# In[28]:
genre_data.head()
# In[29]:
year_data.head()
# In[30]:

# utworzenie okna programu
root=Tk()
root.title('Music Recommendation App') # nazwa programu (wyświetlanie na oknie programu)
root.iconbitmap('play.ico') # ustawienie ikony okna programu

welcome = Label(root, text="Podaj nazwę utworu oraz jego rok wydania: ") # instrukcje
welcome.grid(row=0, column=0)


# In[31]:

def clust_and_vis():
    # klastry
    global song_cluster_pipeline
    song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=8, verbose=True))], verbose=False)

    X = data.select_dtypes(np.number)
    global number_cols
    number_cols = list(X.columns)
    song_cluster_pipeline.fit(X)
    song_cluster_labels = song_cluster_pipeline.predict(X)
    data['cluster_label'] = song_cluster_labels

    # wizualizacja klastrów metodą PCA
    pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
    song_embedding = pca_pipeline.fit_transform(X)
    projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
    projection['title'] = data['name']
    projection['cluster'] = data['cluster_label']

    data['cluster_label'].value_counts()

    fig = px.scatter(projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'])
    fig.show()


# In[32]:

# spotipy
cid = '6795df3cd8eb4cc9acec7bc34230ee6a'
secret = '6cb74b9c803c4415bd1919062cc118ab'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)


# In[33]:

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit','instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

def get_song_data(song, spotify_data):
    
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name']) & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    
    except IndexError:
        return find_song(song['name'], song['year'])

def get_mean_vector(song_list, spotify_data):
    
    song_vectors = []
    
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)  
    
    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)

def flatten_dict_list(dict_list):
    
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
            
    return flattened_dict

def recommend_songs( song_list, spotify_data, n_songs=10):
    clust_and_vis()
    
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)
    
    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')


# In[34]:

songs = []

def display_added_data():
    s_name = s_name_entry.get()
    s_name_label.config(text= s_name)

    s_artist = s_artist_entry.get()
    s_artist_label.config(text= s_artist)

    songs.append({ 'name' : s_name, 'year' : int(s_artist)}) # dodawanie utworów do listy

    display = Label(root, text="**** DODANE UTWORY ****", font = 9) 
    display.grid(row=5, column=0)

    # wyświetlanie wpisanych utworów
    for i in range(len(songs)):
        song = Label(root, text=str(songs[i]['name']) + "    -    " + str(songs[i]['year']))
        song.grid(row=i+7, column=0)

def get_songs():
    # opisuje pole - input utworu
    s_name_des = Label(root, text="Utwór: ")
    s_name_des.grid(row=1, column=0)

    #opisuje pole - input wykonawcy
    s_artist_des = Label(root, text="Rok: ")
    s_artist_des.grid(row=2, column=0)

    # pole - input utworu
    global s_name_entry
    s_name_entry = Entry(root, width=40)
    s_name_entry.grid(row=1, column=1)

    #pole - input wykonawcy
    global s_artist_entry
    s_artist_entry = Entry(root, width=40)
    s_artist_entry.grid(row=2, column=1)

    # output utworu
    global s_name_label
    s_name_label = Label(root, text="")

    # output wykonawcy
    global s_artist_label
    s_artist_label = Label(root, text="")

def display_recommended_data():
    recommend_songs(songs, data)
    recommended_songs = recommend_songs(songs,  data)

    display = Label(root, text="**** POLECANE UTWORY ****", font = 9) 
    display.grid(row=5, column=1)

    for i in range(len(recommended_songs)):
            song = Label(root, text=(str(recommended_songs[i]['name']) + "    -    " + str(recommended_songs[i]['artists']) +"       " + str(recommended_songs[i]['year'])))
            song.grid(row=i+7, column=1)


# In[35]:

get_songs()

# przycisk do wyświetlenia wpisu
add_button = Button(root, text="Dodaj", command=display_added_data, width=15)
add_button.grid(row=1, column=3)

# przycisk do wyświetlenia rekomendacji
recommend_button = Button(root, text="Wyświetl rekomendacje", command=display_recommended_data)
recommend_button.grid(row=2, column=3)


root.mainloop()


# In[ ]:





import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def stem(text):
    y=[]

    for i in text.split():
        y.append(ps.stem(i))

    return " ".join(y)

def recommend(movie):
    movie_list = df[df['title'].str.contains(movie)]
    if len(movie_list):
        movie_idx = movie_list.index[0]
        distances = similarity[movie_idx]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]

        print('Recommendantions for {0} :\n'.format(movie_list.iloc[0]['title']))
        for i in movies_list:
            print(df.iloc[i[0]].title)
    else:
        return "No movies found. Please check your input"

df = pd.read_csv("movies_dataset.csv")
df.head()

ps = PorterStemmer()

for index,row in df.iterrows():
    df.loc[index, 'tags'] = stem(row['tags'])

cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(df['tags']).toarray()

similarity = cosine_similarity(vectors)

recommend('The Matrix')

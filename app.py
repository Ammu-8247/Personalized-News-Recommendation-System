import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# load dataset
data = pd.read_csv("data.csv")

# combine features
data['features'] = data['news_title'] + " " + data['category']

# convert text to numbers
vectorizer = CountVectorizer()
matrix = vectorizer.fit_transform(data['features'])

# similarity calculation
similarity = cosine_similarity(matrix)

def recommend(news_title):
    index = data[data['news_title'] == news_title].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print("Recommended News:\n")
    for i in scores[1:4]:
        print(data.iloc[i[0]]['news_title'])

# input
recommend("AI changing healthcare")

# import libraries (you may add additional imports but you may not have to)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from zipfile import ZipFile
from urllib.request import Request, urlopen, urlretrieve
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
if not os.path.exists("book-crossings.zip"):
    url = "https://cdn.freecodecamp.org/project-data/books/book-crossings.zip"
    req = Request(
        url=url, 
        headers={"User-Agent": "Mozilla/5.0"}
    )

    webpage = urlopen(req)

    with open("book-crossings.zip","wb") as output:
        output.write(webpage.read())

    with ZipFile("book-crossings.zip", "r") as zFile:
        zFile.extractall()

books_filename = "BX-Books.csv"
ratings_filename = "BX-Book-Ratings.csv"
users_filename = "BX-Users.csv"

# Import csv data into dataframes
df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=["isbn", "title", "author"],
    usecols=["isbn", "title", "author"],
    dtype={"isbn": "str", "title": "str", "author": "str"})

df_ratings = pd.read_csv(
    ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=["user", "isbn", "rating"],
    usecols=["user", "isbn", "rating"],
    dtype={"user": "int32", "isbn": "str", "rating": "float32"})

df_users = pd.read_csv(
    users_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=["user", "location", "age"],
    usecols=["user", "location", "age"],
    dtype={"user": "int32", "location": "str", "age": "str"})

# Filtering dataframes
user_ratings = df_ratings['user'].value_counts()
book_ratings = df_ratings['isbn'].value_counts()

df_books.dropna(inplace=True)
df_ratings.dropna(inplace=True)

df_ratings = df_ratings[(~df_ratings['user'].isin(user_ratings[user_ratings < 200].index)) & (~df_ratings['isbn'].isin(book_ratings[book_ratings < 100].index))]

df_ratings = df_ratings.pivot_table(index=['user'],columns=['isbn'],values='rating').fillna(0).T
df_ratings.index = df_ratings.join(df_books.set_index('isbn'))['title'].sort_index()

model = NearestNeighbors(metric='cosine')
model.fit(df_ratings.values)

def get_recommends(book: str = ""):
    try:
        title = df_ratings.loc[book]

    except KeyError as e:
        print('Type an existing book, ', e, 'does not exist')

        return

    distance, indice = model.kneighbors([title.values], n_neighbors=6)

    recommended_books = [book, pd.DataFrame({'title': df_ratings.iloc[indice[0]].index.values, 'distance': distance[0]}).sort_values(by='distance', ascending=False).head(5).values]
    
    return recommended_books

books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(books)

def test_book_recommendation():
  test_pass = True
  recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
  if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
    test_pass = False
  recommended_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
  recommended_books_dist = [0.8, 0.77, 0.77, 0.77]
  for i in range(2): 
    if recommends[1][i][0] not in recommended_books:
      test_pass = False
    if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:
      test_pass = False
  if test_pass:
    print("You passed the challenge! 🎉🎉🎉🎉🎉")
  else:
    print("You havn't passed yet. Keep trying!")

test_book_recommendation()

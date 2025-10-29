from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import pandas as pd
import uvicorn

app = FastAPI(title="Book Recommendation System", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class Book(BaseModel):
    id: int
    title: str
    author: str
    genre: str
    year: int
    rating: float
    pages: int

class BookWithSimilarity(Book):
    similarity_score: float

class RecommendationResponse(BaseModel):
    source_book: Book
    recommendations: List[BookWithSimilarity]

books_data = { 'id': list(range(1, 51)), 'title': [ 'The Great Gatsby', 'To Kill a Mockingbird', '1984', 'Pride and Prejudice', 'The Catcher in the Rye', 'Animal Farm', 'Lord of the Flies', 'Brave New World', 'The Hobbit', 'Fahrenheit 451', 'Jane Eyre', 'Wuthering Heights', 'The Lord of the Rings', 'Harry Potter and the Sorcerer\'s Stone', 'The Da Vinci Code', 'The Alchemist', 'Life of Pi', 'The Kite Runner', 'The Book Thief', 'The Chronicles of Narnia', 'Catch-22', 'Slaughterhouse-Five', 'The Bell Jar', 'One Hundred Years of Solitude', 'Beloved', 'The Handmaid\'s Tale', 'The Road', 'The Shining', 'Gone Girl', 'The Girl with the Dragon Tattoo', 'The Hunger Games', 'Divergent', 'The Fault in Our Stars', 'Eleanor & Park', 'The Perks of Being a Wallflower', 'Looking for Alaska', 'Paper Towns', 'The Giver', 'Ender\'s Game', 'Foundation', 'Dune', 'Neuromancer', 'Snow Crash', 'The Hitchhiker\'s Guide to the Galaxy', 'Good Omens', 'American Gods', 'The Name of the Wind', 'The Way of Kings', 'Mistborn', 'A Game of Thrones' ], 'author': [ 'F. Scott Fitzgerald', 'Harper Lee', 'George Orwell', 'Jane Austen', 'J.D. Salinger', 'George Orwell', 'William Golding', 'Aldous Huxley', 'J.R.R. Tolkien', 'Ray Bradbury', 'Charlotte Brontë', 'Emily Brontë', 'J.R.R. Tolkien', 'J.K. Rowling', 'Dan Brown', 'Paulo Coelho', 'Yann Martel', 'Khaled Hosseini', 'Markus Zusak', 'C.S. Lewis', 'Joseph Heller', 'Kurt Vonnegut', 'Sylvia Plath', 'Gabriel García Márquez', 'Toni Morrison', 'Margaret Atwood', 'Cormac McCarthy', 'Stephen King', 'Gillian Flynn', 'Stieg Larsson', 'Suzanne Collins', 'Veronica Roth', 'John Green', 'Rainbow Rowell', 'Stephen Chbosky', 'John Green', 'John Green', 'Lois Lowry', 'Orson Scott Card', 'Isaac Asimov', 'Frank Herbert', 'William Gibson', 'Neal Stephenson', 'Douglas Adams', 'Terry Pratchett & Neil Gaiman', 'Neil Gaiman', 'Patrick Rothfuss', 'Brandon Sanderson', 'Brandon Sanderson', 'George R.R. Martin' ], 'genre': [ 'Classic', 'Classic', 'Dystopian', 'Romance', 'Classic', 'Dystopian', 'Classic', 'Dystopian', 'Fantasy', 'Dystopian', 'Classic', 'Classic', 'Fantasy', 'Fantasy', 'Mystery', 'Fiction', 'Fiction', 'Fiction', 'Historical Fiction', 'Fantasy', 'Classic', 'Classic', 'Classic', 'Magical Realism', 'Historical Fiction', 'Dystopian', 'Post-Apocalyptic', 'Horror', 'Thriller', 'Thriller', 'Dystopian', 'Dystopian', 'Young Adult', 'Young Adult', 'Young Adult', 'Young Adult', 'Young Adult', 'Dystopian', 'Science Fiction', 'Science Fiction', 'Science Fiction', 'Science Fiction', 'Science Fiction', 'Science Fiction', 'Fantasy', 'Fantasy', 'Fantasy', 'Fantasy', 'Fantasy', 'Fantasy' ], 'year': [ 1925, 1960, 1949, 1813, 1951, 1945, 1954, 1932, 1937, 1953, 1847, 1847, 1954, 1997, 2003, 1988, 2001, 2003, 2005, 1950, 1961, 1969, 1963, 1967, 1987, 1985, 2006, 1977, 2012, 2005, 2008, 2011, 2012, 2013, 1999, 2005, 2008, 1993, 1985, 1951, 1965, 1984, 1992, 1979, 1990, 2001, 2007, 2010, 2006, 1996 ], 'rating': [ 4.2, 4.5, 4.4, 4.3, 3.8, 4.3, 3.9, 4.1, 4.6, 4.2, 4.3, 4.1, 4.7, 4.6, 3.9, 3.9, 3.9, 4.3, 4.6, 4.2, 4.0, 4.1, 4.0, 4.1, 4.2, 4.3, 4.1, 4.3, 4.2, 4.3, 4.5, 4.2, 4.3, 4.2, 4.3, 4.2, 4.0, 4.3, 4.3, 4.3, 4.2, 4.2, 4.1, 4.4, 4.3, 4.3, 4.6, 4.6, 4.5, 4.5 ], 'pages': [ 180, 324, 328, 432, 277, 112, 224, 288, 310, 194, 500, 416, 1178, 309, 454, 208, 319, 371, 552, 767, 453, 275, 244, 417, 324, 311, 287, 447, 415, 465, 374, 487, 313, 328, 213, 221, 305, 179, 324, 255, 688, 271, 440, 224, 288, 465, 662, 1007, 541, 694 ] }
df = pd.DataFrame(books_data)

# Feature engineering for KNN
def prepare_features(df):
    genre_map = {g: i for i, g in enumerate(df['genre'].unique())}
    df['genre_encoded'] = df['genre'].map(genre_map)
    df['year_normalized'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
    features = df[['genre_encoded', 'year_normalized', 'rating', 'pages']].values
    return features, genre_map

features, genre_map = prepare_features(df)

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Train KNN model
knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
knn.fit(features_scaled)

@app.get("/")
async def root():
    return {
        "message": "Book Recommendation System API",
        "version": "1.0.0",
        "endpoints": {
            "books": "/api/books",
            "book_detail": "/api/books/{book_id}",
            "recommendations": "/api/recommendations/{book_id}",
            "search": "/api/search?q=query",
            "genres": "/api/genres",
            "books_by_genre": "/api/books/genre/{genre}"
        }
    }
books_data = {
    'id': list(range(1, 101)),
    'title': [
        # 1–50 (Original)
        'The Great Gatsby', 'To Kill a Mockingbird', '1984', 'Pride and Prejudice',
        'The Catcher in the Rye', 'Animal Farm', 'Lord of the Flies', 'Brave New World',
        'The Hobbit', 'Fahrenheit 451', 'Jane Eyre', 'Wuthering Heights',
        'The Lord of the Rings', 'Harry Potter and the Sorcerer\'s Stone', 'The Da Vinci Code',
        'The Alchemist', 'Life of Pi', 'The Kite Runner', 'The Book Thief',
        'The Chronicles of Narnia', 'Catch-22', 'Slaughterhouse-Five', 'The Bell Jar',
        'One Hundred Years of Solitude', 'Beloved', 'The Handmaid\'s Tale',
        'The Road', 'The Shining', 'Gone Girl', 'The Girl with the Dragon Tattoo',
        'The Hunger Games', 'Divergent', 'The Fault in Our Stars', 'Eleanor & Park',
        'The Perks of Being a Wallflower', 'Looking for Alaska', 'Paper Towns',
        'The Giver', 'Ender\'s Game', 'Foundation', 'Dune', 'Neuromancer',
        'Snow Crash', 'The Hitchhiker\'s Guide to the Galaxy', 'Good Omens',
        'American Gods', 'The Name of the Wind', 'The Way of Kings', 'Mistborn',
        'A Game of Thrones',
        # 51–100 (New additions)
        'A Clash of Kings', 'A Storm of Swords', 'A Feast for Crows', 'A Dance with Dragons',
        'The Lies of Locke Lamora', 'Red Rising', 'Golden Son', 'Morning Star',
        'Ready Player One', 'Armada', 'Project Hail Mary', 'The Martian',
        'Artemis', 'Contact', 'Hyperion', 'The Fall of Hyperion',
        'Children of Time', 'Children of Ruin', 'Leviathan Wakes', 'Caliban\'s War',
        'Abaddon\'s Gate', 'Cibola Burn', 'Nemesis Games', 'Babylon\'s Ashes',
        'Persepolis Rising', 'Tiamat\'s Wrath', 'Leviathan Falls', 'Snowpiercer',
        'The Maze Runner', 'The Scorch Trials', 'The Death Cure', 'The Kill Order',
        'The Fifth Season', 'The Obelisk Gate', 'The Stone Sky', 'Circe',
        'The Song of Achilles', 'The Silence of the Lambs', 'Red Dragon',
        'Hannibal', 'Hannibal Rising', 'It', 'Pet Sematary', 'Doctor Sleep',
        'Salem\'s Lot', 'Carrie', 'Misery', '11/22/63', 'The Institute', 'Fairy Tale'
    ],
    'author': [
        # 1–50
        'F. Scott Fitzgerald', 'Harper Lee', 'George Orwell', 'Jane Austen',
        'J.D. Salinger', 'George Orwell', 'William Golding', 'Aldous Huxley',
        'J.R.R. Tolkien', 'Ray Bradbury', 'Charlotte Brontë', 'Emily Brontë',
        'J.R.R. Tolkien', 'J.K. Rowling', 'Dan Brown', 'Paulo Coelho',
        'Yann Martel', 'Khaled Hosseini', 'Markus Zusak', 'C.S. Lewis',
        'Joseph Heller', 'Kurt Vonnegut', 'Sylvia Plath', 'Gabriel García Márquez',
        'Toni Morrison', 'Margaret Atwood', 'Cormac McCarthy', 'Stephen King',
        'Gillian Flynn', 'Stieg Larsson', 'Suzanne Collins', 'Veronica Roth',
        'John Green', 'Rainbow Rowell', 'Stephen Chbosky', 'John Green',
        'John Green', 'Lois Lowry', 'Orson Scott Card', 'Isaac Asimov',
        'Frank Herbert', 'William Gibson', 'Neal Stephenson', 'Douglas Adams',
        'Terry Pratchett & Neil Gaiman', 'Neil Gaiman', 'Patrick Rothfuss',
        'Brandon Sanderson', 'Brandon Sanderson', 'George R.R. Martin',
        # 51–100
        'George R.R. Martin', 'George R.R. Martin', 'George R.R. Martin', 'George R.R. Martin',
        'Scott Lynch', 'Pierce Brown', 'Pierce Brown', 'Pierce Brown',
        'Ernest Cline', 'Ernest Cline', 'Andy Weir', 'Andy Weir',
        'Andy Weir', 'Carl Sagan', 'Dan Simmons', 'Dan Simmons',
        'Adrian Tchaikovsky', 'Adrian Tchaikovsky', 'James S.A. Corey', 'James S.A. Corey',
        'James S.A. Corey', 'James S.A. Corey', 'James S.A. Corey', 'James S.A. Corey',
        'James S.A. Corey', 'James S.A. Corey', 'James S.A. Corey', 'Jacques Lob',
        'James Dashner', 'James Dashner', 'James Dashner', 'James Dashner',
        'N.K. Jemisin', 'N.K. Jemisin', 'N.K. Jemisin', 'Madeline Miller',
        'Madeline Miller', 'Thomas Harris', 'Thomas Harris',
        'Thomas Harris', 'Thomas Harris', 'Stephen King', 'Stephen King', 'Stephen King',
        'Stephen King', 'Stephen King', 'Stephen King', 'Stephen King', 'Stephen King', 'Stephen King'
    ],
    'genre': [
        # 1–50
        'Classic', 'Classic', 'Dystopian', 'Romance', 'Classic', 'Dystopian',
        'Classic', 'Dystopian', 'Fantasy', 'Dystopian', 'Classic', 'Classic',
        'Fantasy', 'Fantasy', 'Mystery', 'Fiction', 'Fiction', 'Fiction',
        'Historical Fiction', 'Fantasy', 'Classic', 'Classic', 'Classic',
        'Magical Realism', 'Historical Fiction', 'Dystopian', 'Post-Apocalyptic',
        'Horror', 'Thriller', 'Thriller', 'Dystopian', 'Dystopian', 'Young Adult',
        'Young Adult', 'Young Adult', 'Young Adult', 'Young Adult', 'Dystopian',
        'Science Fiction', 'Science Fiction', 'Science Fiction', 'Science Fiction',
        'Science Fiction', 'Science Fiction', 'Fantasy', 'Fantasy', 'Fantasy',
        'Fantasy', 'Fantasy', 'Fantasy',
        # 51–100
        'Fantasy', 'Fantasy', 'Fantasy', 'Fantasy', 'Fantasy', 'Science Fiction', 'Science Fiction', 'Science Fiction',
        'Science Fiction', 'Science Fiction', 'Science Fiction', 'Science Fiction', 'Science Fiction', 'Science Fiction',
        'Science Fiction', 'Science Fiction', 'Science Fiction', 'Science Fiction', 'Science Fiction', 'Science Fiction',
        'Science Fiction', 'Science Fiction', 'Science Fiction', 'Science Fiction',
        'Science Fiction', 'Science Fiction', 'Science Fiction', 'Dystopian',
        'Dystopian', 'Dystopian', 'Dystopian', 'Dystopian', 'Fantasy', 'Fantasy', 'Fantasy',
        'Fantasy', 'Fantasy', 'Thriller', 'Thriller', 'Thriller', 'Thriller',
        'Horror', 'Horror', 'Horror', 'Horror', 'Horror', 'Horror', 'Horror', 'Horror', 'Horror'
    ],
    'year': [
        # 1–50
        1925, 1960, 1949, 1813, 1951, 1945, 1954, 1932, 1937, 1953,
        1847, 1847, 1954, 1997, 2003, 1988, 2001, 2003, 2005, 1950,
        1961, 1969, 1963, 1967, 1987, 1985, 2006, 1977, 2012, 2005,
        2008, 2011, 2012, 2013, 1999, 2005, 2008, 1993, 1985, 1951,
        1965, 1984, 1992, 1979, 1990, 2001, 2007, 2010, 2006, 1996,
        # 51–100
        1998, 2000, 2005, 2011, 2006, 2014, 2015, 2016,
        2011, 2015, 2021, 2014, 2017, 1985, 1989, 1990,
        2015, 2019, 2011, 2012, 2013, 2014, 2015, 2016,
        2017, 2019, 2021, 1982, 2009, 2010, 2011, 2012,
        2015, 2016, 2017, 2018, 2011, 1988, 1981, 1999,
        2006, 1986, 1983, 2013, 1975, 1974, 1987, 2011, 2019, 2022
    ],
    'rating': [
        # 1–50
        4.2, 4.5, 4.4, 4.3, 3.8, 4.3, 3.9, 4.1, 4.6, 4.2,
        4.3, 4.1, 4.7, 4.6, 3.9, 3.9, 3.9, 4.3, 4.6, 4.2,
        4.0, 4.1, 4.0, 4.1, 4.2, 4.3, 4.1, 4.3, 4.2, 4.3,
        4.5, 4.2, 4.3, 4.2, 4.3, 4.2, 4.0, 4.3, 4.3, 4.3,
        4.2, 4.2, 4.1, 4.4, 4.3, 4.3, 4.6, 4.6, 4.5, 4.5,
        # 51–100
        4.6, 4.7, 4.5, 4.4, 4.5, 4.3, 4.4, 4.5,
        4.4, 3.8, 4.6, 4.5, 4.1, 4.3, 4.4, 4.3,
        4.4, 4.5, 4.4, 4.4, 4.4, 4.4, 4.5, 4.4,
        4.4, 4.4, 4.5, 4.0, 4.0, 4.1, 4.1, 4.0,
        4.6, 4.6, 4.7, 4.4, 4.5, 4.3, 4.2, 4.1,
        4.1, 4.2, 4.4, 4.3, 4.1, 4.0, 4.2, 4.3, 4.4, 4.5
    ],
    'pages': [
        # 1–50
        180, 324, 328, 432, 277, 112, 224, 288, 310, 194,
        500, 416, 1178, 309, 454, 208, 319, 371, 552, 767,
        453, 275, 244, 417, 324, 311, 287, 447, 415, 465,
        374, 487, 313, 328, 213, 221, 305, 179, 324, 255,
        688, 271, 440, 224, 288, 465, 662, 1007, 541, 694,
        # 51–100
        768, 973, 1061, 1056, 752, 382, 464, 518,
        374, 349, 496, 387, 305, 432, 482, 517,
        600, 608, 582, 595, 539, 598, 600, 608,
        640, 720, 720, 296, 375, 384, 325, 327,
        512, 448, 464, 393, 378, 367, 348, 536,
        564, 1138, 416, 531, 439, 199, 320, 849, 576, 608
    ]
}


@app.get("/api/books", response_model=List[Book])
async def get_books():
    """Get all books"""
    books = df.to_dict('records')
    return books

@app.get("/api/books/{book_id}", response_model=Book)
async def get_book(book_id: int):
    """Get a specific book by ID"""
    book = df[df['id'] == book_id].to_dict('records')
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return book[0]

@app.get("/api/recommendations/{book_id}", response_model=RecommendationResponse)
async def get_recommendations(
    book_id: int,
    n: int = Query(default=5, ge=1, le=20, description="Number of recommendations")
):
    """Get book recommendations based on KNN algorithm"""
    try:
        book_idx = df[df['id'] == book_id].index[0]
    except IndexError:
        raise HTTPException(status_code=404, detail="Book not found")
    
    distances, indices = knn.kneighbors([features_scaled[book_idx]], n_neighbors=n + 1)
    
    recommended_indices = indices[0][1:]
    recommended_distances = distances[0][1:]
    
    recommendations = []
    for idx, dist in zip(recommended_indices, recommended_distances):
        book = df.iloc[idx].to_dict()
        book['similarity_score'] = round((1 / (1 + dist)) * 100, 2)
        recommendations.append(book)
    
    return {
        'source_book': df[df['id'] == book_id].iloc[0].to_dict(),
        'recommendations': recommendations
    }

@app.get("/api/search", response_model=List[Book])
async def search_books(q: str = Query(..., min_length=1, description="Search query")):
    """Search books by title or author"""
    query = q.lower()
    results = df[
        df['title'].str.lower().str.contains(query) | 
        df['author'].str.lower().str.contains(query)
    ].to_dict('records')
    return results

@app.get("/api/genres", response_model=List[str])
async def get_genres():
    """Get all unique genres"""
    return df['genre'].unique().tolist()

@app.get("/api/books/genre/{genre}", response_model=List[Book])
async def get_books_by_genre(genre: str):
    """Get all books in a specific genre"""
    books = df[df['genre'].str.lower() == genre.lower()].to_dict('records')
    if not books:
        raise HTTPException(status_code=404, detail="No books found for this genre")
    return books

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

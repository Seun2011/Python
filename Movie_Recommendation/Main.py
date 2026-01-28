import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from colorama import init, Fore
import time
import sys

init(autoreset=True)

def load_data(file_path='imdb_top_1000.csv'):
    try:
        df = pd.read_csv(file_path)

        genre_col = 'Genre' if 'Genre' in df.columns else 'genre'
        overview_col = 'Overview' if 'Overview' in df.colunms else 'overview'
        rating_col = 'IMDB_Rating' if 'IMDB_Rating'
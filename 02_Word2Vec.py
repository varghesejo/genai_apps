# 9. Use a prebuilt Word2vec model to find the embedding of the given word. 

# Create the python env for executing the code
# conda create --prefix <path with env name> pip ipykernel 

# Activate the python environment and install the required dependency packages
# conda activate <env_name>
# pip/conda install gensim pandas nltk

from gensim.models import Word2Vec
from gensim.test.utils import common_texts
import gensim.downloader

#from gensim.models import Word2Vec, KeyedVectors
import pandas as pd
import nltk
#import kagglehub

def word2vec_embedding1():
    model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
    model.save("word2vec.model")
    model = Word2Vec.load("word2vec.model")
    vector = model.wv['computer'] # get numpy vector of a word
    print("\nEmbeddings for the word 'computer':\n", vector)

    similar = model.wv.most_similar('computer', topn=10)  # get other similar words
    print("\nMost simillar word for 'computer':")
    for item in similar:
        print("\t", item)

    vector1 = model.wv['user'] 
    print("\nEmbeddings for the word 'user':\n", vector1)

    similar1 = model.wv.most_similar('user', topn=10)  # get other similar words
    print("\nMost simillar word for 'user'")
    for item in similar1:
        print("\t", item)
    
def list_the_pretrained_models():
    gensim_pretrained_models = list(gensim.downloader.info()['models'].keys())
    print("\n\nList of pre-trained models in gensim are:")
    for model in gensim_pretrained_models:
        print("\t", model)
    print("-----------------------\n")

def word2vec_pretrained_model_google_news():
    print("\n\nUsing Word2Vec Pretrained model google news...")
    word2vec_vectors = gensim.downloader.load('word2vec-google-news-300')

    word2vec_vectors.most_similar('google', topn=10)

    sims = word2vec_vectors["king"] - word2vec_vectors["man"] + word2vec_vectors["woman"]
    print("\nMost similar word for 'king' - 'man' + 'woman':")
    for item in word2vec_vectors.most_similar([sims]):
        print("\t", item)   
    #word2vec_vectors.most_similar([sims])

    sims = word2vec_vectors["Messi"] - word2vec_vectors["football"] + word2vec_vectors["cricket"]
    print("\n\nMost similar word for 'Messi' - 'football' + 'cricket':")
    for item in word2vec_vectors.most_similar([sims]):
        print("\t", item)
    #word2vec_vectors.most_similar([sims])

def word2vec_embedding2():
    nltk.download('punkt')

    # Get the dataset from Kaggle
    # path = kagglehub.dataset_download("rootuser/worldnews-on-reddit")
    # print("Path to dataset files:", path)
    # Will be downloaded to : C:\Users\<User>\.cache\kagglehub\datasets\rootuser\worldnews-on-reddit\versions\1

    filename = "reddit_worldnews_start_to_2016-11-22.csv"
    print(f"\n\n Using the dataset file: {filename}\n")
    df = pd.read_csv('reddit_worldnews_start_to_2016-11-22.csv')

    print("\tShape of the data set:", df.shape)

    newsTitles = df['title'].values
    #print("\n\nNews Titles: \n", newsTitles)

    newsVec = [nltk.word_tokenize(title) for title in newsTitles]
    #newsVec

    model = Word2Vec(newsVec, min_count=5, window=5, vector_size=100, workers=4)

    vector = model.wv['man']
    print("\nEmbeddings for the word 'man':\n", vector)
    sims = model.wv.most_similar('man', topn=10)
    print("\nMost simillar word for 'man':")
    for item in sims:
        print("\t", item)
    
    vector = model.wv['police']
    print("\nEmbeddings for the word 'police':\n", vector)
    sims1 = model.wv.most_similar('police', topn=10)
    print("\nMost simillar word for 'police':")
    for item in sims1:
        print("\t", item)
    

def main():
    list_the_pretrained_models()
    word2vec_embedding1()
    word2vec_embedding2()
    word2vec_pretrained_model_google_news()

if __name__ == "__main__":
    print("\n9. Use a prebuilt Word2vec model to find the embedding of the given word. ")
    print("---------------------------------------------------------------------------")
    main()
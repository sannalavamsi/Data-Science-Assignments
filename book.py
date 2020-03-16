
import pandas as pd
book_data = pd.read_csv("C:\\Users\\dell\\Downloads\\Recommended Systems\\books1.csv",encoding='ISO-8859-1')
book_data.shape
book_data.columns
book_data=book_data.drop(book_data.iloc[:,0:1],axis=1)

from sklearn.feature_extraction.text import TfidfVectorizer #term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus

# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words="english")    #taking stop words from tfid vectorizer 


tfidf_matrix = tfidf.fit_transform(book_data["Book.Author"])   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape 

# with the above matrix we need to find the 
# similarity score
# There are several metrics for this
# such as the euclidean, the Pearson and 
# the cosine similarity scores

# For now we will be using cosine similarity matrix
# A numeric quantity to represent the similarity 
# between 2 movies 
# Cosine similarity - metric is independent of 
# magnitude and easy to calculate 

# cosine(x,y)= (x.y⊺)/(||x||.||y||)

from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix)

# creating a mapping of book title to index number 
book_index = pd.Series(book_data.index, index=book_data['Book.Author']).drop_duplicates()



def get_book_recommendations(book_author,topN=10):
    
    
    #topN = 10
    # Getting the book index using its title 
    book_id = book_index["Amy Tan"]
    
    
    cosine_scores = list(enumerate(cosine_sim_matrix[book_id]))
    
     
    cosine_scores = sorted(cosine_scores,key=lambda x:x[1],reverse = True)
    
    # Get the scores of top 10 most similar books
    cosine_scores_10 = cosine_scores[0:topN+1]
    
    # Getting the book index 
    book_idx  =  [i[0] for i in cosine_scores_10]
    book_scores =  [i[1] for i in cosine_scores_10]
    
    # Similar books and scores
    book_similar_show = pd.DataFrame(columns=["author","Score"])
    book_similar_show["author"] = book_data.loc[book_idx,"book_author"]
    book_similar_show["Score"] = book_scores
    book_similar_show.reset_index(inplace=True)  
    book_similar_show.drop(["index"],axis=1,inplace=True)
    print (book_similar_show)
 

    
# Enter your book and number of book to be recommended 
get_book_recommendations("Amy Tan",topN=15)

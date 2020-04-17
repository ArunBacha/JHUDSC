import pyspark.sql.functions as sql_func
from pyspark.sql.types import *
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.evaluation import RegressionEvaluator
spark = SparkSession.builder.appName("CollaborativeRecommendationSystem").master("local").enableHiv

df_books = spark.sql("select * from bigdata_project.books")
df_books.show(5)
df_books_sub = df_books.select("book_id","authors","title","original_publication_year","language_co
df_books_sub.show(5)
df_ratings = spark.sql("select * from bigdata_project.ratings")
df_book_ratings = df_ratings.join(df_books_sub, on = ['book_id'])
df_book_ratings.show(5)

max_rating = float(df_book_ratings.describe("average_rating").filter("summary = 'max'").select("ave
min_rating = float(df_book_ratings.describe("average_rating").filter("summary = 'min'").select("ave
print(max_rating)
print(min_rating)

 #Some modifications to improve our model
def modify_predictions(prediction):
    if prediction < min_rating:
        return min_rating
    elif prediction > max_rating:
        return max_rating
    else :
        return prediction
 
(training_ratings,test_ratings) = df_book_ratings.randomSplit([0.8,0.2])
colloborative_als=ALS(maxIter=20,regParam=0.09,rank=25,userCol="user_id",itemCol="book_id",ratingCo
colloborative_model =colloborative_als.fit(training_ratings)
 
coll_predictions = colloborative_model.transform(test_ratings)
coll_predictions.show(5)

from pyspark.sql.functions import udf
udf_modify_predictions = udf(modify_predictions, FloatType())
coll_predictions = coll_predictions.withColumn("prediction",udf_modify_predictions(coll_predictions
coll_predictions.show(5)

coll_evaluator = RegressionEvaluator(metricName="rmse", labelCol="average_rating", predictionCol="prediction")
Coll_RMSE = coll_evaluator.evaluate(coll_predictions)
print("Root-mean-square error = " + str(Coll_RMSE))

test_ratings.show(5)

user_predictions = test_ratings.filter(test_ratings['user_id']== 439).select(['book_id','user_id'])
user_predictions.show()

user_predicted_ratings = colloborative_model.transform(user_predictions)
user_predicted_ratings.show()
userRecommendations = colloborative_model.recommendForAllUsers(5)
userRecommendations.show(10)
bookRecommendations = colloborative_model.recommendForAllItems(5)
bookRecommendations.show(10)


df_tags = spark.sql("select * from bigdata_project.tags")
df_book_tags = spark.sql("select  * from bigdata_project.book_tags")
tags_df = df_tags.toPandas()


import re
import string
tags_df['tag_name'] = tags_df['tag_name'].apply(lambda x : re.sub(r'\W+', '', x).lstrip(string.digits)

 
import numpy as np
tags_df['tag_name'] .replace('', np.nan, inplace=True)
tags_df.dropna(subset=['tag_name'], inplace=True)

 
book_tags = book_tags.join(tags_df, on = "tag_id", how= "inner",lsuffix='_left', rsuffix='_right')
book_tags = book_tags.drop(['tag_id_right', 'tag_id_left'], axis = 1)

 
book_tags = book_tags.groupby(['goodreads_book_id'])['tag_name'].apply(' '.join).reset_index()


books_df = df_books.toPandas()
books_df = pd.merge(books_df, book_tags, left_on='book_id', right_on='goodreads_book_id', how='inne
books_df.head()

books_df['corpus'] = (pd.Series(books_df[['authors', 'tag_name', 'title']].fillna('').values.tolist()).str.join(' '))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words=['english', 'currently
tfidf = tf.fit_transform(books_df['corpus'].head(10000))
cos_tags_auths = linear_kernel(tfidf, tfidf)

book_titles = books_df['title']
indices = pd.Series(books_df.index, index=books_df['title'])
# Function that get book recommendations based on the cosine similarity score of books tags
def tags_authors_recommendation(title):
    idx = indices[title]
    simulated_scores = list(enumerate(cos_tags_auths[idx]))
    simulated_scores = sorted(simulated_scores, key=lambda x: x[1], reverse=True)
    simulated_scores = simulated_scores[1:21]
    book_df_indices = [i[0] for i in simulated_scores]
    return book_titles.iloc[book_df_indices]

book_recom = tags_authors_recommendation('The Hobbit').head(5).to_frame()
book_recom.head()
 
book_recom['book_id'] = book_recom.index
book_recom['user_id'] = 1169
book_recom['title'].head()

book_recom = book_recom[['user_id','book_id']]

book_recom =  spark.createDataFrame(book_recom)
type(book_recom)
 

predicted_ratings_hybrid = colloborative_model.transform(book_recom)
predicted_ratings_hybrid.show()
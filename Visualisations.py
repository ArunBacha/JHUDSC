from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pyspark.sql.functions import avg
from pyspark.sql import SQLContext
from pyspark.sql.types import StringType
spark = SparkSession.builder.appName("res_app").master("local").enableHiveSupport().getOrCreate()
df = spark.sql("select * from bigdata_project.books")
df.printSchema()
df.show(5)
df.describe().show()
books_df = df.toPandas()
df= spark.sql("select * from bigdata_project.ratings")
df.printSchema()
df.show(5)
df.describe().show()
ratings_df = df.toPandas()
df= spark.sql("select * from bigdata_project.book_tags")
df.printSchema()
df.show(5)
df.describe().show()
book_tags_df = df.toPandas()
df= spark.sql("select * from bigdata_project.tags")
df.printSchema()
df.show(5)
df.describe().show()
tags_df = df.toPandas()
df= spark.sql("select * from bigdata_project.books_to_read")
df.printSchema()
df.show(5)
df.describe().show()
books_to_read_df = df.toPandas()
plt.hist(books_df["average_rating"], bins = 5)
plt.title("Histogram showing the number of books for each rating")
plt.xlabel('Rating')
plt.xticks(range(1,6,1))
plt.show()


book_tags_df.head()

books_df = books_df.iloc[:, :-2]
books_df.head()

ratings_df.head()

tags_df.head()

books_to_read_df.head()

def rating_aggregated(row):
    if(row["average_rating"] <= 1.0):
        return 1
    elif (row["average_rating"] > 1.0 and row["average_rating"] <= 2.0):
        return 2
    elif (row["average_rating"] > 2.0 and row["average_rating"] <= 3.0):
        return 3
    elif (row["average_rating"] > 3.0 and row["average_rating"] <= 4.0):
        return 4
    elif (row["average_rating"] > 4.0 and row["average_rating"] <= 5.0):
        return 5


books_df["Rounded_Rating"] = books_df.apply(rating_aggregated, axis = 1)

books_df.head()

rating_per_user = ratings_df.groupby('user_id').user_id.apply(lambda id : len(id))
rating_per_user.head()

#How the users are rating different books
rating_per_user.describe()

#Number of Books in each language
books_df.groupby('language_code').language_code.apply(lambda id : len(id))

#Top 10 Rated books
books_df[["title","average_rating"]].nlargest(10,['average_rating'])

def fifty_years(row):
    if (row["original_publication_year"] >= 1750.0 and row["original_publication_year"] <1800.0):
        return "1750_1800"
    elif (row["original_publication_year"] > 1800.0 and row["original_publication_year"] <=1850.0):
        return "1800_1850"
    elif (row["original_publication_year"] > 1850.0 and row["original_publication_year"] <=1900.0):
        return "1850_1900"
    elif (row["original_publication_year"] > 1900.0 and row["original_publication_year"] <=1950.0):
        return "1900_1950"
    elif (row["original_publication_year"] > 1950.0 and row["original_publication_year"] <=2000.0):
        return "1950_2000"
    elif (row["original_publication_year"] > 2000.0 and row["original_publication_year"] <=2050.0):
        return "2000_2050"


books_df["fifty_years"] = books_df.apply(fifty_years, axis = 1)

#plot 1
Year_Rating_Box, ax1 = plt.subplots(1,1)
Year_Rating_Box.suptitle('Violin plot showing the rating and year published', fontsize=14)
ax1 = sns.violinplot(x="fifty_years", y="average_rating",data=books_df, inner="quart", linewidth=1.3,ax = ax1)
ax1.set_xlabel("Published Year",size = 12,alpha=0.8)
ax1.set_ylabel("Average Rating",size = 12,alpha=0.8)
plt.show()


#Plot2
ax = plt.figure(figsize=(15,10))
books_df.dropna(0, inplace=True)
sns.set_context('paper')
ax =sns.jointplot(x="average_rating",y='work_text_reviews_count', kind='scatter', data= books_df[['work_text_reviews_count','average_rating']])
ax.set_axis_labels("Average Rating", "Text Review Count")
plt.title("Relation Between Review count and Rating", loc = "left")


#plot3

from wordcloud import WordCloud
book_popularity = books_to_read_df.merge(books_df[["book_id", "title"]], on = "book_id")
df_book_popularity= book_popularity["title"]
plt.subplots(figsize=(10,10))
wc = WordCloud( background_color='white',width=512,height=412).generate(" ".join(df_book_popularity))
plt.imshow(wc)
plt.axis('off')
plt.show()


#plot4
book_languages = books_df['language_code'].value_counts()
book_languages_df = pd.DataFrame(book_languages)
book_languages_df = book_languages_df.reset_index()
book_languages_df = book_languages_df.rename(columns={"index": "lang_code", "language_code": "counts"}
book_languages_df = book_languages_df.sort_values(['counts'], ascending=False).head(5)
labels = book_languages_df.lang_code
counts = book_languages_df.counts
explode = (.05, 0, 0, 0,0)
fig, ax = plt.subplots(figsize = (10,10))
ax.pie(counts,explode=explode, labels=labels, autopct='%1.f%%', shadow=False, startangle=90)
ax.axis('equal')

#plot5

filter_list = ["eng","fre","spa","ger"]
books_lang=books_df[books_df.language_code.isin(filter_list)]
books_lang["language_code"].value_counts()
sns.set(context='notebook', style='whitegrid')
sns.boxplot(x="language_code", y="average_rating", data=books_lang, whis="range", palette="vlag")
#Add in points to show each observation
sns.swarmplot(x="language_code", y="average_rating", data=books_lang, linewidth=0)
plt.title('Distribution of Books & Rating Categories', fontsize=15)
plt.show()

#plot6
Popular_books = books_df.groupby('authors')['title'].count().reset_index().sort_values('title', ascend
plt.figure(figsize=(16,16))
sbx = sns.barplot(Popular_books['title'], Popular_books.index)
sbx.set_title("Top 10 authors with most number of book writings")
sbx.set_xlabel("No of Books Published")
sbx.set_ylabel("Authors with most popular writings")

#plot7
f,ax = plt.subplots(figsize=(7,7))
sns.heatmap(books_df[["books_count","original_publication_year","average_rating", "ratings_count","wor
plt.show()

#plot8
plt.figure(figsize=(20,10))
sns.boxplot(y=books_df.average_rating, x=books_df.language_code)
plt.show()


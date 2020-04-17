from pyspark.sql import SparkSession
import  pandas as pd
import  numpy as np
from pymongo import MongoClient
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf
spark = SparkSession.builder.appName('demography mapper').getOrCreate()
df_books = spark.read.format("com.mongodb.spark.sql.DefaultSource").option("spark.mongodb.input.uri", "mongodb://localhost:27017/BigData_Project.Books").load()
df_book_tags = spark.read.format("com.mongodb.spark.sql.DefaultSource").option("spark.mongodb.input.uri", "mongodb://localhost:27017/BigData_Project.Book_Tags").load()
df_Ratings = spark.read.format("com.mongodb.spark.sql.DefaultSource").option("spark.mongodb.input.uri", "mongodb://localhost:27017/BigData_Project.Ratings").load()
df_Tags = spark.read.format("com.mongodb.spark.sql.DefaultSource").option("spark.mongodb.input.uri", "mongodb://localhost:27017/BigData_Project.Tags").load()
df_Books_To_Read = spark.read.format("com.mongodb.spark.sql.DefaultSource").option("spark.mongodb.input.uri", "mongodb://localhost:27017/BigData_Project.Books_To_Read").load()
df_books.show(n=2)
df_book_tags.show(n=2)
df_Ratings.show(n=2)
df_Tags.show(n=2)
df_Books_To_Read.show(n=2)

df_books = df_books.drop("_ID")
df_book_tags = df_book_tags.drop("_ID")
df_Ratings = df_Ratings.drop("_ID")
df_Tags = df_Tags.drop("_ID")
df_Books_To_Read = df_Books_To_Read.drop("_ID")

df_books.write.format("csv").save("hdfs://sandbox-hdp.hortonworks.com:8020/user/maria_dev/tmp/books")

df_book_tags.write.format("csv").save("hdfs://sandbox-hdp.hortonworks.com:8020/user/maria_dev/tmp/book_tags")

df_Ratings.write.format("csv").save("hdfs://sandbox-hdp.hortonworks.com:8020/user/maria_dev/tmp/Ratings")

df_Tags.write.format("csv").save("hdfs://sandbox-hdp.hortonworks.com:8020/user/maria_dev/tmp/Tags")

df_Books_To_Read.write.format("csv").save("hdfs://sandbox-hdp.hortonworks.com:8020/user/maria_dev/tmp/Books_To_Read")


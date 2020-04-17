The main aim of this project is to build a hybrid recommendation system for book readers. 
The Dataset we have obtained is from goodreads which have 10k books and the users, tags related to the book and ratings provided by the user to the books they have read. 

Approach: 

We have downloaded the data in the form of CSV files and loaded the csv files into MongoDB collections. 
The Code for this part can be found in BigDataDataInsertion.py

Once we have the data in MongoDB collections, we  will start a spark session and load the data from MongoDB collections and write it to HDFS.  We have setup MongoDB on HDP 2.6 version. We have also downloaded few jars to setup a connection between MongoDB and spark session. 
We intiate a spark session, fetch the data into spark session and write it to HDFS
Code can be found in the file Spark_Mongo_To_Hive.py

We have created External Tables on top of the csv files we have written to HDFS in the above step

Once the data is available in Hive tables, we performed exploratory data analysis on the dataset.
The EDA can be found in the file visualisations.py

After EDA, we have implemented hybrid recommendation system which would be based on both content based recommendations and colloborative filtering approach. Implementation details would be found in the code component MLImplementation.py


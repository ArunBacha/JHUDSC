from cassandra.cluster import Cluster
from cassandra.cluster import  ExecutionProfile, EXEC_PROFILE_DEFAULT
from cassandra.policies import WhiteListRoundRobinPolicy, DowngradingConsistencyRetryPolicy, ConsistencyLevel
from cassandra.query import tuple_factory


#Connecting to Cassandra DB
profile = ExecutionProfile(
    load_balancing_policy=WhiteListRoundRobinPolicy(['127.0.0.1']),
    retry_policy=DowngradingConsistencyRetryPolicy(),
    consistency_level=ConsistencyLevel.LOCAL_QUORUM,
    serial_consistency_level=ConsistencyLevel.LOCAL_SERIAL,
    request_timeout=15,
    row_factory=tuple_factory
)
cluster = Cluster(execution_profiles={EXEC_PROFILE_DEFAULT: profile})

#Establishing a session
session = cluster.connect()

#Creating a keyspace 
session.execute("""CREATE KEYSPACE IF NOT EXISTS Project_Data WITH REPLICATION = { 'class' : 'SimpleStrategy', 'replication_factor' : 1 };""")

#Creating a table to hold the train images in the folder
session.execute("""CREATE TABLE Project_Data.train_images (
                         file_name text,
                         image blob,
                         PRIMARY KEY (file_name)
                       );""")

#Creating a table to hold the test images in the folder
session.execute("""CREATE TABLE Project_Data.train_annotations (
                         file_name text,
                         file_data text,
                         PRIMARY KEY (file_name)
                       );""")

#Creating a table to hold the test annotations
session.execute("""CREATE TABLE Project_Data.test_annotations (
                         file_name text,
                         file_data text,
                         PRIMARY KEY (file_name)
                       );""")

#Creating a table to hold the train annotations
session.execute("""CREATE TABLE Project_Data.train_annotations (
                         file_name text,
                         file_data text,
                         PRIMARY KEY (file_name)
                       );""")

#Shutting down the connection cluster
cluster.shutdown()
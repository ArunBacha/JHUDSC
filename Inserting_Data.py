# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 17:41:34 2020

@author: arunk
"""

from cassandra.cluster import Cluster
from cassandra.cluster import  ExecutionProfile, EXEC_PROFILE_DEFAULT
from cassandra.policies import WhiteListRoundRobinPolicy, DowngradingConsistencyRetryPolicy, ConsistencyLevel
from cassandra.query import tuple_factory
import os
from os.path import isfile, join
import base64

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

session = cluster.connect()

wd = "D:/Trent/Project/Data/RarePlanes_train_PS-RGB_tiled/PS-RGB_tiled"
strCQL = "INSERT INTO Project_Data.train_images(file_name,image) VALUES (?,?)"
pStatement = session.prepare(strCQL)

os.chdir(wd)
for file in os.listdir(wd):
    if file.endswith(".png"):
        #Extract the image and then convert the image and insert into the cassandra database
        image = open(file, "rb")
        image_read = image.read()
        image_64_encode = base64.encodestring(image_read)
        session.execute(pStatement,[file,image_64_encode])

wd = "D:/Trent/Project/Data/RarePlanes_test_PS-RGB_tiled/PS-RGB_tiled"
os.chdir(wd)
strCQL = "INSERT INTO Project_Data.test_images(file_name,image) VALUES (?,?)"
pStatement = session.prepare(strCQL)

for file in os.listdir(wd):
    if file.endswith(".png"):
        print(file)
        #Extract the image and then convert the image and insert into the cassandra database
        image = open(file, "rb")
        image_read = image.read()
        image_64_encode = base64.encodestring(image_read)
        session.execute(pStatement,[file,image_64_encode])

wd = "D:/Trent/Project/Data/RarePlanes_test_PS-RGB_tiled/PS-RGB_tiled"
os.chdir(wd)
strCQL = "INSERT INTO Project_Data.test_annotations(file_name,file_data) VALUES (?,?)"
pStatement = session.prepare(strCQL)

for file in os.listdir(wd):
    json = open(file, encoding ="cp437")
    json_content = json.read()
    session.execute(pStatement, [file,json_content])

wd = "D:/Trent/Project/Data/RarePlanes_train_PS-RGB_tiled/PS-RGB_tiled"
os.chdir(wd)
strCQL = "INSERT INTO Project_Data.train_annotations(file_name,file_data) VALUES (?,?)"
pStatement = session.prepare(strCQL)

for file in os.listdir(wd):
    json = open(file,mode="r",encoding="cp437")
    json_content = json.read()
    session.execute(pStatement, [file,json_content])

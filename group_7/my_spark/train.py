#!/usr/bin/python
#  -*- coding: utf-8 -*-
from pyspark.sql import SparkSession
from pyspark import SparkFiles

spark = SparkSession.builder \
    .appName("SparkSessionExample") \
    .master("spark://master:7077") \
    .config("spark.some.config.option", "some-value") \
    .config('spark.debug.maxToStringFields', '100') \
    .config('spark.shuffle.service.enabled', 'true') \
    .getOrCreate()

# 获取SparkContext实例对象
sc = spark.sparkContext

# 获取推荐数据
sc.addFile("data/ml-100k/u_proc.data")
sc.addFile("/data/ml-100k/u.item")

# 在集群中的每个节点上访问该文件
dataFilePath = SparkFiles.get("u_proc.data")
rdd = sc.textFile(dataFilePath)
#rdd = sc.textFile("/tmp/data/ml-100k/u_proc.data")

# 转换RDD格式
rawRatings = rdd.map(lambda line: line.split("\t")[:3])
ratingsRDD = rawRatings.map(lambda x: (x[0], x[1], x[2]))

# 构建itemid2name数据
itemRDD = sc.textFile(SparkFiles.get("u.item"))
itemid2name = itemRDD.map(lambda line: line.split("|")).map(lambda a: (float(a[0]), a[1])).collectAsMap()

# 训练
from pyspark.mllib.recommendation import ALS
model = ALS.train(ratingsRDD, 12, 20, 0.01)


from pymongo import MongoClient as MC
from pymongo.errors import ConnectionFailure, OperationFailure

# MongoDB Connection
try:
    client = MC('124.70.27.99', 27017)
    db = client.admin
    db.authenticate("admin", "123456", mechanism='SCRAM-SHA-1')
    userRec = db.userRec
except ConnectionFailure as e:
    print("Could not connect to MongoDB: %s" % e)

# 构建用户列表并生成推荐
useridlist = range(1, 944)

# 使用try-except块确保数据库操作不会导致程序崩溃
try:
    for userid in useridlist:
        if userid % 50 == 0:
            print('Processing userid:', userid)

        recp = model.recommendProducts(int(userid), 10)
        userrecinfo = {
            'userid': userid,
            'recbooks': [{'itemid': str(p[1]), 'itemname': itemid2name.get(p[1], 'Unknown'), 'rate': str(p[2])} for p in recp]
        }

        # 写入MongoDB数据库
        userRec.insert_one(userrecinfo)
except OperationFailure as e:
    print("An error occurred: %s" % e)

finally:
    # 关闭数据库连接
    client.close()




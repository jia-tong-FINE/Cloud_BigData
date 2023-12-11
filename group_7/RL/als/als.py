import os
import redis
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType, FloatType


def process_row(r):
    pv_count = r.pv if r.pv else 0.0
    fav_count = r.fav if r.fav else 0.0
    cart_count = r.cart if r.cart else 0.0
    buy_count = r.buy if r.buy else 0.0

    pv_score = 0.2 * pv_count if pv_count <= 20 else 4.0
    fav_score = 0.4 * fav_count if fav_count <= 20 else 8.0
    cart_score = 0.6 * cart_count if cart_count <= 20 else 12.0
    buy_score = 1.0 * buy_count if buy_count <= 20 else 20.0

    rating = pv_score + fav_score + cart_score + buy_score
    return r.userId, r.cateId, rating


def recall_cate_by_cf(partition):
    pool = redis.ConnectionPool(host=host, port=port)
    client = redis.Redis(connection_pool=pool)
    for row in partition:
        client.hset("recall_cate", row.userId, [i.cateId for i in row.recommendations])


if __name__ == "__main__":

    PYSPARK_PYTHON = "/miniconda2/envs/py365/bin/python"
    JAVA_HOME = '/root/bigdata/jdk'
    SPARK_HOME = "/root/bigdata/spark"
    os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
    os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_PYTHON
    os.environ['JAVA_HOME'] = JAVA_HOME
    os.environ["SPARK_HOME"] = SPARK_HOME

    SPARK_APP_NAME = "als"
    SPARK_URL = "spark://192.168.19.137:7077"

    conf = SparkConf()  # 创建spark config对象
    config = (
        ("spark.app.name", SPARK_APP_NAME),
        ("spark.executor.memory", "6g"),
        ("spark.master", SPARK_URL),
        ("spark.executor.cores", "4"),
    )

    conf.setAll(config)
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    # df = spark.read.csv("/data/behavior_log.csv", header=True)
    # df.show()
    # df.printSchema()
    schema = StructType([
        StructField("userId", IntegerType()),
        StructField("timestamp", LongType()),
        StructField("btag", StringType()),
        StructField("cateId", IntegerType()),
        StructField("brandId", IntegerType())
    ])

    behavior_log_df = spark.read.csv("./data/behavior_log.csv", header=True, schema=schema)
    print(behavior_log_df.show())
    print(behavior_log_df.count())

    cate_count_df = behavior_log_df \
        .groupBy(behavior_log_df.userId, behavior_log_df.cateId).pivot("btag", ["pv", "fav", "cart", "buy"]).count()
    cate_count_df.printSchema()

    # spark.sparkContext.setCheckpointDir("./checkPoint/")

    # 构建结构对象
    schema = StructType([
        StructField("userId", IntegerType()),
        StructField("cateId", IntegerType()),
        StructField("pv", IntegerType()),
        StructField("fav", IntegerType()),
        StructField("cart", IntegerType()),
        StructField("buy", IntegerType())
    ])

    # cate_count_df = spark.read.csv("/preprocessing_dataset/cate_count.csv", header=True, schema=schema)
    cate_count_df.printSchema()
    cate_count_df.first()

    cate_rating_df = cate_count_df.rdd.map(process_row).toDF(["userId", "cateId", "rating"])

    als = ALS(userCol='userId', itemCol='cateId', ratingCol='rating', checkpointInterval=5)
    model = als.fit(cate_rating_df)

    ret = model.recommendForAllUsers(3)

    host = "192.168.19.137"
    port = 6379

    # 召回到redis
    result.foreachPartition(recall_cate_by_cf)
    result.count()

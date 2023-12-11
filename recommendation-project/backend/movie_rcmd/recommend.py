from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.recommendation import ALS, ALSModel

MONGODB_URI = 'mongodb://node1:27017,node2:27017,node3:27017,node4:27017/rcmd_db?replicaSet=mongorepl'

def ALS_recommend():
    spark = SparkSession.builder \
        .master('yarn') \
        .appName('Movie Recommendation') \
        .config('spark.mongodb.input.uri', MONGODB_URI) \
        .config('spark.mongodb.output.uri', MONGODB_URI) \
        .config('spark.yarn.archive', 'hdfs:///home/spark_jars.zip') \
        .getOrCreate()
    # spark = SparkSession.builder\
    #     .appName('Movie Recommendation') \
    #     .config('spark.mongodb.input.uri', MONGODB_URI) \
    #     .config('spark.mongodb.output.uri', MONGODB_URI) \
    #     .getOrCreate()
    sc = spark.sparkContext

    # generated by django, i.e. our user - (user, movie, rating)
    dj_ratings = spark.read.format('mongo') \
        .option('collection', 'movie_rcmd_rating') \
        .load() \
        .drop('_id', 'id') \
        .withColumnRenamed('user_id', 'userId') \
        .withColumnRenamed('movie_id', 'movieId')
    dj_ratings = dj_ratings \
        .withColumn('rating', dj_ratings.rating.cast(DoubleType()))
    # dj_ratings.show()
    num_dj_ratings = dj_ratings.count()

    # movielens datasets - (userId, movieId, rating)
    ml_ratings = spark.read.csv("hdfs:///home/ratings.csv", inferSchema=True, header=True) \
        .drop('timestamp')
    # ml_ratings = spark.read.csv("hdfs:///home/ratings.csv", schema='userId INT, movieId INT, rating DOUBLE, timestamp INT', header=True) \
    ml_ratings = ml_ratings.withColumn('userId', ml_ratings.userId + num_dj_ratings)
    # ml_ratings.show()

    # spark.stop()
    # return
    # ml_ratings_rdd = ml_ratings.rdd.map(lambda x: (x[0] + num_dj_ratings, x[1], x[2]))
    # ml_ratings = spark.createDataFrame(ml_ratings_rdd, ml_ratings.schema)

    all_ratings = dj_ratings.unionByName(ml_ratings)
    # all_ratings.show()

    als = ALS(maxIter=5, regParam=0.01, userCol='userId', itemCol='movieId', ratingCol='rating')
    als_model: ALSModel = als.fit(all_ratings)


    # users = spark.createDataFrame(sc.parallelize([str(i) for i in range(1, 11)]), ['userId'])
    # users = ratings.select('userId').distinct().limit(10).rdd.map(lambda x: (x[0] + 1,))
    # users = spark.createDataFrame(users, ['userId'])
    recommendation_rdd = als_model.recommendForUserSubset(dj_ratings, 20).rdd \
        .flatMap(lambda row: [(row[0], movie_id, weight) for movie_id, weight in row[1]])

    recommendation_rdd = recommendation_rdd.zipWithIndex() \
        .map(lambda row: (row[1], *row[0]))
    result = spark.createDataFrame(recommendation_rdd, ['id', 'user_id', 'movie_id', 'rating'])
    result.show()
    result.write.format('mongo').mode('overwrite') \
        .option('collection', 'movie_rcmd_moviercmd') \
        .save()

    spark.stop()

if __name__ == '__main__':
    ALS_recommend()
from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS,Rating
from pyspark.mllib.recommendation import MatrixFactorizationModel

spark_context = SparkContext("local", "Test")
model = MatrixFactorizationModel.load(spark_context, 'als-model')
# movie_dict=csv_map_key("../data/movie.csv")

for item in model.recommendProducts(1,5):
    print(item.product)
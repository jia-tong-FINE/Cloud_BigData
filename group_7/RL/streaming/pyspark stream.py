import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="streaming_data")
    parser.add_argument("--data_log", type=str, default="behavior")
    parser.add_argument("--input_path", type=str, default="../als/data")
    parser.add_argument("--output_path", type=str, default="./output")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # process behavior data
    if args.data_log == 'behavior':

        print("Application--behavior Started ...")
        spark = SparkSession\
            .builder\
            .appName('behavior Streaming - CSV')\
            .master('local[*]')\
            .getOrCreate()

        input_behaviors = StructType([
            StructField("userId", IntegerType()),
            StructField("timestamp", LongType()),
            StructField("btag", StringType()),
            StructField("cateId", IntegerType()),
            StructField("brandId", IntegerType())
        ])

        stream_behaviors_df = spark \
            .readStream \
            .format("csv") \
            .option("header", "true") \
            .schema(input_behaviors) \
            .load(path=args.input_path) # path

        stream_behaviors_df.printSchema()

        uid_num = stream_behaviors_df.groupBy("userId").count().count()
        btag_class_count = stream_behaviors_df.groupBy("btag").count().collect()
        before_dropna = stream_behaviors_df.count()
        after_dropna = stream_behaviors_df.dropna().count()

        print('uid_num', uid_num)
        print('btag_class_count', btag_class_count)
        print('exist_na_value', before_dropna, after_dropna)

        # row_column transform

        #user - cate
        user_cate_df = stream_behaviors_df\
            .groupBy(stream_behaviors_df.userId, stream_behaviors_df.cateId)\
            .pivot("btag",["pv", "fav", "cart", "buy"])\
            .count()

        # user - brand
        user_brand_df = stream_behaviors_df\
            .groupBy(stream_behaviors_df.userId, stream_behaviors_df.brandId)\
            .pivot("btag", ["pv", "fav", "cart", "buy"])\
            .count()

        user_cate_df.write.csv("./output/user_cate.csv", header=True)
        brand_count_df.write.csv("./output/user_brand.csv", header=True)

        stream_df_query = stream_behaviors_df \
            .writeStream \
            .format("json") \
            .start()

        stream_df_query.awaitTermination()

        print("Completed.")

    # process raw data
    if args.data_log == 'raw':

        print("Application--rawdata Started ...")
        spark = SparkSession \
            .builder \
            .appName('rawdata Streaming - CSV') \
            .master('local[*]') \
            .getOrCreate()

        input_rawdata = StructType([
            StructField("user", StringType()),
            StructField("timestamp", StringType()),
            StructField("adgroup_id", StringType()),
            StructField("pid", StringType()),
            StructField("nonclk", StringType()),
            StructField("clk", StringType()),
        ])

        stream_raw_df = spark \
            .readStream \
            .format("csv") \
            .option("header", "true") \
            .schema(input_rawdata) \
            .load(path=args.input_path)  # path

        stream_raw_df.printSchema()

        print("sample_num:", stream_raw_df.count())
        print("user_num:", stream_raw_df.groupBy("user").count().count())
        print("ad_num:", stream_raw_df.groupBy("adgroup_id").count().count())
        print("pid_class_count:", stream_raw_df.groupBy("pid").count().collect())
        print("clk:", stream_raw_df.groupBy("clk").count().collect())

        unify_raw_df = stream_raw_df. \
            withColumn("user", stream_raw_df.user.cast(IntegerType())).withColumnRenamed("user", "userId"). \
            withColumn("time_stamp", stream_raw_df.time_stamp.cast(LongType())).withColumnRenamed("time_stamp", "timestamp"). \
            withColumn("adgroup_id", stream_raw_df.adgroup_id.cast(IntegerType())).withColumnRenamed("adgroup_id", "adgroupId"). \
            withColumn("pid", stream_raw_df.pid.cast(StringType())). \
            withColumn("nonclk", stream_raw_df.nonclk.cast(IntegerType())). \
            withColumn("clk", stream_raw_df.clk.cast(IntegerType()))

        unify_raw_df.printSchema()

        # one hot encode pid

        stringindexer = StringIndexer(inputCol='pid', outputCol='pid_feature')
        encoder = OneHotEncoder(dropLast=False, inputCol='pid_feature', outputCol='pid_value')

        pipeline = Pipeline(stages=[stringindexer, encoder])
        pipeline_model = pipeline.fit(unify_raw_df)
        new_df = pipeline_model.transform(unify_raw_df)

        new_df.write.csv("./output/rawdata_new.csv", header=True)

        stream_df_query = unify_raw_df \
            .writeStream \
            .format("json") \
            .start()

        stream_df_query.awaitTermination()

        print("Completed.")

    # process user data
    if args.data_log == 'user':

        print("Application--user Started ...")
        spark = SparkSession \
            .builder \
            .appName('user Streaming - CSV') \
            .master('local[*]') \
            .getOrCreate()

        input_user = StructType([
            StructField("userId", IntegerType()),
            StructField("cms_segid", IntegerType()),
            StructField("cms_group_id", IntegerType()),
            StructField("final_gender_code", IntegerType()),
            StructField("age_level", IntegerType()),
            StructField("pvalue_level", IntegerType()),
            StructField("shopping_level", IntegerType()),
            StructField("occupation", IntegerType()),
            StructField("new_user_class_level", IntegerType())
        ])

        stream_user_df = spark \
            .readStream \
            .format("csv") \
            .option("header", "true") \
            .schema(input_user) \
            .load(path=args.input_path)  # path

        stream_user_df.printSchema()

        print("cms_segid: ", stream_user_df.groupBy("cms_segid").count().count())
        print("cms_group_id: ", stream_user_df.groupBy("cms_group_id").count().count())
        print("final_gender_code: ", stream_user_df.groupBy("final_gender_code").count().count())
        print("age_level: ", stream_user_df.groupBy("age_level").count().count())
        print("shopping_level: ", stream_user_df.groupBy("shopping_level").count().count())
        print("occupation: ", stream_user_df.groupBy("occupation").count().count())

        # see null value
        #stream_user_df.groupBy("pvalue_level").count().show()
        #stream_user_df.groupBy("new_user_class_level").count().show()

        t_count = stream_user_df.count()
        pl_na_count = t_count - stream_user_df.dropna(subset=["pvalue_level"]).count()
        print("pvalue_level:", pl_na_count, "空值占比：%0.2f%%" % (pl_na_count / t_count * 100))
        nul_na_count = t_count - stream_user_df.dropna(subset=["new_user_class_level"]).count()
        print("new_user_class_level:", nul_na_count, "空值占比：%0.2f%%" % (nul_na_count / t_count * 100))

        # fill null value using random forest

        train_data = stream_user_df.dropna(subset=['pvalue_level']).rdd.map(
            lambda r: LabeledPoint(r.pvalue_level - 1,
                                   [r.cms_segid, r.cms_group_id, r.final_gender_code, r.age_level, r.shopping_level, r.occupation]))

        model = RandomForest.trainClassifier(train_data, 3, {}, 5)
        pl_na_df = stream_user_df.na.fill(-1).where("pvalue_level=-1")


        def row(r):
            return r.cms_segid, r.cms_group_id, r.final_gender_code, r.age_level, r.shopping_level, r.occupation

        rdd = pl_na_df.rdd.map(row)
        predicts = model.predict(rdd)

        print(predicts.take(10))

        temp = predicts.map(lambda x: int(x)).collect()
        dff = pl_na_df.toPandas()

        dff["pvalue_level"] = np.array(temp) + 1

        new_user_df = stream_user_df.dropna(subset=["pvalue_level"]).unionAll(spark.createDataFrame(dff, schema=input_user))

        new_user_df.write.csv("./output/user_new.csv", header=True)

        stream_df_query = stream_user_df \
            .writeStream \
            .format("json") \
            .start()

        stream_df_query.awaitTermination()

        print("Completed.")





























import os
import sys
sys.path.append(os.pardir)
import warnings
warnings.filterwarnings("ignore")
import argparse
import time
import numpy as np
import pandas as pd
from pyspark import SparkConf
from pyspark.sql import SparkSession



def parse_args():
    parser = argparse.ArgumentParser(description="run_prepare_data")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def bucket_age(age):
    if age < 30:
        return 1
    elif age < 40:
        return 2
    elif age < 50:
        return 3
    else:
        return 4


if __name__ == "__main__":
    args = parse_args()
    print(vars(args))
    np.random.seed(args.seed)
    start_time = time.perf_counter()

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

    user_feat = spark.read.csv("resources/user.csv", header=None).toPandas()
    item_feat = spark.read.csv("resources/item.csv", header=None).toPandas()
    behavior = spark.read.csv("resources/user_behavior.csv", header=None).toPandas()

    user_feat.columns = range(user_feat.columns.size)
    item_feat.columns = range(item_feat.columns.size)
    behavior.columns = range(behavior.columns.size)

    user_feat.rename(columns={0: "user", 1: "sex", 2:"age", 3:"pur_power"})
    item_feat.rename(columns={0: "item", 1: "category", 2: "shop", 3: "brand"})
    behavior.rename(columns={0: "user", 1: "item", 2: "behavior", 3: "time"})

    behavior = behavior.sort_values(by="time").reset_index(drop=True)
    behavior = behavior.drop_duplicates(subset=["user", "item", "behavior"])

    user_counts = behavior.groupby("user")[["user"]].count().rename(
        columns={"user": "count_user"}
    ).sort_values("count_user", ascending=False)
    # sample users with short and long sequences
    short_users = np.array(
        user_counts[
            (user_counts.count_user > 5) & (user_counts.count_user <= 50)
        ].index
    )
    long_users = np.array(
        user_counts[
            (user_counts.count_user > 50) & (user_counts.count_user <= 200)
        ].index
    )
    short_chosen_users = np.random.choice(short_users, 60000, replace=False)
    long_chosen_users = np.random.choice(long_users, 20000, replace=False)
    chosen_users = np.concatenate([short_chosen_users, long_chosen_users])

    behavior = behavior[behavior.user.isin(chosen_users)]
    print(f"n_users: {behavior.user.nunique()}, "
          f"n_items: {behavior.item.nunique()}, "
          f"behavior length: {len(behavior)}")

    # merge with all features
    behavior = behavior.merge(user_feat, on="user")
    behavior = behavior.merge(item_feat, on="item")
    behavior["age"] = behavior["age"].apply(bucket_age)
    behavior = behavior.sort_values(by="time").reset_index(drop=True)
    behavior.to_csv("resources/tianchi.csv", header=None, index=False)
    print(f"prepare data done!, "
          f"time elapsed: {(time.perf_counter() - start_time):.2f}")

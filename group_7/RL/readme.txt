rl 中包含设计的离线reinforcement learning算法，data 存储在rl/resource中，通过spark进行数据预处理，embedding，通过 run_reinforce.py 进行训练，run_bcq 为设计对比的deep Q-learning算法。

als 中包含调用pyspark接口的协同过滤ALS算法，数据存储在als/data中，通过spark进行数据预处理，调用spark.ml进行模型训练

streaming 中包含由Spark streaming 实现的实时监听与计算的功能，通过file streaming监听所需路径的csv文件变化，实时获取新文件，同时通过spark输出聚合数据分析，通过调用spark.ml调用one hot encoder 以及 random forest等实现对新增数据的实时计算与处理

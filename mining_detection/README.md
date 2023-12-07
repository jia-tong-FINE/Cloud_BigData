# 第二组:基于自编码器的挖矿流量检测系统

使用docker-compose，启动对应环境
```
docker-compose up -d
```
将文件传入hadoop,由于流量文件相对较大，该部分直接上传对应的特征提取文件
```
docker exec namenode hdfs dfs -mkdir /input
docker exec namenode hdfs dfs -put /input_files/train.csv /input
```

进行训练过程，使用一部分特征样本进行训练
```
docker exec -it master /bin/bash
cd /python && /opt/bitnami/spark/bin/spark-submit --master spark://master:7077 --py-files /python/ train.py
```

#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
from pymongo import MongoClient as MC
from pymongo.errors import ConnectionFailure, PyMongoError
import os

# 设置命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--userid', type=int, required=True)  # 使userid参数成为必需的
args = parser.parse_args()

try:
    # 连接MongoDB数据库
    client = MC('124.70.27.99', 27017)
    db = client.admin
    db.authenticate("admin", "123456", mechanism='SCRAM-SHA-1')

    userRec = db.userRec
    result = userRec.find_one({'userid': args.userid})

    if result:
        movies = result['recbooks']
        movies.sort(key=lambda x: float(x['rate']), reverse=True)

        # 创建输出目录（如果不存在）
        output_dir = 'data/output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 写入推荐到文件
        with open(os.path.join(output_dir, 'usr_' + str(args.userid) + '_rec.txt'), 'w') as f:
            f.write('对用户 %s 进行推荐：\n' % str(args.userid))
            for movie in movies:
                f.write('%d. 推荐：%s, 推荐指数为: %.2f，在源数据中id为：%s\n' % (
                    movies.index(movie) + 1, movie['itemname'], float(movie['rate']), movie['itemid']))
    else:
        print("没有找到用户ID为 {} 的推荐数据。".format(args.userid))

except ConnectionFailure:
    print("数据库连接失败，请检查您的连接设置和网络。")
except PyMongoError as e:
    print("MongoDB操作出现错误：", e)
finally:
    # 确保MongoDB连接被关闭
    client.close()

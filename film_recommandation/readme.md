# 电影推荐系统部署测试文档

## 部署与启动方法

+ 我们将centos系统打包成tar文件，用于导入docker镜像
  由于tar文件过大，无法直接上传至github,故存放在百度网盘中
  链接：https://pan.baidu.com/s/1s7jU933Eu9xFAhU5N_u7ow?pwd=qq71  提取码：qq71
  下载tar文件后运行命令： `docker import film.tar film` 导入docker镜像
+ 切换至film目录，运行命令： `python3 manage.py runserver 0.0.0.0:8001`
  启动Django服务器
+ 运行命令： `python3 spark.py` 根据用户操作记录生成推荐列表

## 实现效果

+ 注册新用户，登录后进入个人主页
  ![](film/static/images/1.png)
+ 初始浏览列表、收藏列表均为空，故推荐列表为空
  ![](film/static/images/2.png)
  ![](film/static/images/3.png)
  ![](film/static/images/4.png)
+ 用户进行操作后，运行离线推荐spark脚本，生成推荐列表
  ![](film/static/images/5.png)
  ![](film/static/images/6.png)
  ![](film/static/images/7.png)


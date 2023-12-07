from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql import SparkSession
import torch
import torch.nn as nn
import torch.optim as optim

spark = SparkSession.builder.appName("Python Spark RF example").config("spark.some.config.option", "some-value").getOrCreate()

# 加载数据
df = spark.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load("hdfs://namenode:8020/input/train.csv",header=True)

df.dtypes # Return df column names and data types
df.show()  #Display the content of df
df.head()  #Return first n rows

df = df.na.fill(value='')  # 缺失填充值

example = df.select(['param', 'label'])


tokenizer = Tokenizer(inputCol="param", outputCol="words")
wordsData = tokenizer.transform(example)

# 使用HashingTF将文本转换为特征向量
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(wordsData)


# 计算TF-IDF
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# 查看计算结果
rescaledData.select("features").show(truncate=False)

features = []
for vec in rescaledData.collect():
    print("text: ", vec.param)
    print("label: ", vec.label)
    print("vector: ", list(vec.features.toArray()))
    features.append(vec.features.toArray())
    print("=================================")
print(type(features[0]))

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

input_size = len(features[0])
print(type(features[0]))
print(input_size)
hidden_size = 8  # 可根据需要调整
model = Autoencoder(input_size, hidden_size).double()#.cuda()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练 Autoencoder 模型
num_epochs = 1000
for epoch in range(num_epochs):
    for data in features:
        data = torch.from_numpy(data)
        optimizer.zero_grad()
        reconstructions = model(data) # .cuda()
        loss = criterion(reconstructions, data) # .cuda()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

torch.save(model, 'model.pth')
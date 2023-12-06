# 第三组：基于云原生与分布式计算的文本生成服务

## 提交内容
* distributed-llama2-server：主仓库，完整的基础实现 `by sirui, xingyu` (xingyu负责streaming的实现，见[#commit_1](https://github.com/Siritao/distributed-llama2-server/commit/edf36c0d6530c5c40d763eff7252375c2106fc46)，[#commit_2](https://github.com/Siritao/distributed-llama2-server/commit/9c5f95aa9a7c481ed58d27408fc60b947dc93cf1))
* continuous_batching：改进1-连续批处理服务 `by xingyu`
* quantization：改进2-量化相关 `by xinhao`

## 快速部署
详见主仓库的`readme`

注：由于llama2模型非常大（>10G）且使用需要授权，所以我们仅在服务器上尝试了打包镜像、一键部署，没有直接提供给用户，主仓库提供的Dockerfile不包含下载模型这一部分

## 其他
* llm serving是近期非常火的方向，我们花了大量时间与精力调研梳理现有方案并进行测试（在老师提供的服务器上有部署好的环境），具体设计时，也尝试了不同的方案，踩了许多坑，提交的代码为最终版本
* 在服务器上已经实现服务的稳定部署，可以随时启动测试（稳定支持多client并发访问，但容器好像会定期清理）
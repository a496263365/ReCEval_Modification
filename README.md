# ReCEval_Modification

本仓库参照[ReCEval论文](https://arxiv.org/abs/2304.10703)提出的Correctness和Informativeness两个维度的指标，
对Agent系统的多轮对话trajectory进行评测。

## 与原论文实现的区别

1. 使用一个大模型替换了原论文中繁琐复杂的多个小模型
2. 原论文中评测一个简单CoT段落，而这里压缩每段对话为一个summary语句，评测多轮对话下的连贯程度

## 仓库结构说明

```
|--data  #存放得到的trajectory数据
|--|--raw_data  #得到的trajectories.jsonl文件
|--|--summary_data  #对每个trajectory的每段话进行总结，一个trajectory得到一个.json文件，文件名为instance_id
|--|--receval_result  #经过计算得到的指标值
|--llm_clients  #定义访问LLM的API
|--|--BaseLLMClient.py  #存放访问地址和Secret Key信息
|--|--DSV3Client.py  #Deepseek-v3
|--tools  #乱七八糟的可能有用的小工具
|--|--list_all_usable_models.py
|--trajectory_summary.py  #对trajectories.jsonl进行总结，得到一系列.json文件
|--receval_modification.py  #对summary.json调用LLM进行打分评测
```

## How to Use
直接运行`trajectory_summary.py`和`receval_modification.py`中的`main`函数

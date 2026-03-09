# vkgqa_eval
一为大语言模型（LLM）在知识图谱问答（KGQA）、视觉问答（VQA）及多步推理任务设计的自动化评估工具。
1. 项目简介 
本仓库提供了一套用于评估大语言模型（LLM）在知识图谱问答（KGQA）和视觉问答（VQA）任务中表现的自动化评估工具。它特别针对输出格式不规范、三元组提取以及多跳推理（Multi-hop Reasoning）等复杂场景进行了优化。

2. 核心特性
突出代码的“卖点”，即它比简单的 string.equal() 强在哪里：

鲁棒的文本标准化：集成 VQAEval，支持自动缩写还原（如 "don't" -> "dont"）、数字转换（"one" -> "1"）和标点过滤。

灵活的三元组解析：能够从模型生成的文本中自动提取 (Subject, Predicate, Object) 结构，支持 JSON、管道符 | 或逗号等多种分隔符。

多维度统计：支持按任务类别（Category）、推理跳数（Hop）以及知识来源（Knowledge）进行细粒度的准确率分析。

容错设计：自动检测输入文件中的列名，兼容多种常见的 JSONL 字段命名。

3. 数据格式要求 (Data Format)

Ground Truth (GT) 文件示例
JSON
{"index": "1", "answer": "Paris", "category": "text", "hop": "1", "knowledge": "123"}
{"index": "2", "answer": [["Albert Einstein", "born in", "Ulm"]], "category": "graph understand triple"}

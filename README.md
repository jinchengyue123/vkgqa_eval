# vkgqa_eval


## 1. 项目简介 (Project Overview)

> **示例：**
> 本仓库提供了一套用于评估大语言模型（LLM）在知识图谱问答（KGQA）和视觉问答（VQA）任务中表现的自动化评估工具。它特别针对输出格式不规范、三元组提取以及多跳推理（Multi-hop Reasoning）等复杂场景进行了优化。

## 2. 核心特性 (Key Features)

* **鲁棒的文本标准化**：集成 `VQAEval`，支持自动缩写还原（如 "don't" -> "dont"）、数字转换（"one" -> "1"）和标点过滤。
* **灵活的三元组解析**：能够从模型生成的文本中自动提取 `(Subject, Predicate, Object)` 结构，支持 JSON、管道符 `|` 或逗号等多种分隔符。
* **多维度统计**：支持按任务类别（Category）、推理跳数（Hop）以及知识来源（Knowledge）进行细粒度的准确率分析。
* **容错设计**：自动检测输入文件中的列名，兼容多种常见的 JSONL 字段命名。

## 3. 数据格式要求 (Data Format)

### Ground Truth (GT) 文件示例

```json
{"index": "1", "answer": "Paris", "category": "text", "hop": "1", "knowledge": "123"}
{"index": "2", "answer": [["Albert Einstein", "born in", "Ulm"]], "category": "graph understand triple"}

```

### Prediction 文件示例

```json
{"index": "1", "prediction": "The answer is Paris."}
{"index": "2", "prediction": "Albert Einstein | born in | Ulm"}

```

## 4. 快速上手 (Quick Start)

提供运行命令示例：

```bash
python evaluate.py \
    --tsv data/ground_truth.jsonl \
    --pred data/model_predictions.jsonl \
    --per-sample-out results/details.jsonl \
    --summary-out results/summary.txt

```

## 5. 评估逻辑说明 (Evaluation Logic)

简要解释几种匹配模式：

* **VQA Match**: 基于核心词提取的宽松匹配，过滤停用词。
* **Triple Exact Match**: 只有当预测的所有三元组（无序）与标准答案完全一致时才判为正确。
* **List Match**: 针对列表类答案（如颜色列表），不考虑顺序，只检查成员一致性。


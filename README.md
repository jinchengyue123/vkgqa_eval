这是一个为你定制的 `README.md` 文档，你可以直接复制以下内容并在你的项目仓库或文件夹中使用。

***

# VKG-QA 评测脚本 (Evaluation Script)

该脚本用于对多模态知识图谱问答（VKG-QA）模型的推理结果进行自动化评测。脚本支持标准 VQA 宽松字符串匹配、三元组精确匹配（Triple Exact Match）以及列表类答案的评估，并能根据任务类型（Category）、推理跳数（Hop）和知识类型（Knowledge）自动生成详细的准确率统计报告。

## ✨ 主要特性 (Features)

* **极简用法**：只需输入一个包含“预测值”和“标准答案”的 JSONL 文件即可一键评测。
* **兼容性强**：自动识别数据中的常见字段名（如 `answer/label`，`prediction/output`），无需严格绑定固定字段。
* **多维度统计**：自动按 `Category`（任务类型）、`Hop`（推理跳数）、`Knowledge`（外部知识）和 `Reasoning Type` 输出详细的 Accuracy 报告。
* **智能 VQA 匹配**：内置标准 VQA 评测逻辑，自动处理大小写、标点符号、缩写（如 `ain't` -> `ain't` 转换处理）和冠词（a/an/the）。
* **三元组无序评估**：针对 `graph understand triple` 任务，支持将字符串解析为 `(head, relation, tail)` 列表，并进行无序 Exact Match 评估。
* **无损数据注入**：在输出带有评测结果的新 JSONL 时，完美保留原始文件的所有字段，仅在每行末尾新增 `"correct": 1/0` 属性。

## 🛠️ 环境依赖 (Dependencies)

* Python 3.7+
* `pandas`

```bash
pip install pandas
```

## 🚀 快速开始 (Quick Start)

### 基本用法

只需传入合并了预测值和真实值的 JSONL 文件：

```bash
python eval_kgqa.py --input your_model_results.jsonl
```
*运行后，将自动在当前目录生成两个文件：*
1.  `your_model_results_evaluated.jsonl`：包含评测结果（附加了 `correct` 字段）的完整数据。
2.  `your_model_results_summary.txt`：包含多维度准确率的纯文本报告。

### 自定义输出路径

如果你想指定输出文件的名称，可以使用可选参数：

```bash
python eval_kgqa.py \
    --input your_model_results.jsonl \
    --out-jsonl ./results/my_evaluated_data.jsonl \
    --out-txt ./results/my_report.txt
```

## 📄 数据格式要求 (Data Format)

输入文件必须是 **JSONL (JSON Lines)** 格式，即每一行都是一个独立的 JSON 对象。

脚本会**自动寻找**以下键值（Key），只要你的数据包含其中之一即可被识别：

* **标准答案 (Ground Truth)**: `gt_answers`, `label`, `ground_truth`, `answer`, `answers`
* **模型预测 (Prediction)**: `prediction`, `pred`, `output`, `model_answer`, `response`

**可选（但强烈建议包含）的分析字段：**
* **任务类型**: `category`, `cat`, `type`, `task_type` (支持细分 `graph understand triple` 等专门逻辑)
* **推理跳数**: `hop`, `hops`, `num_hop`
* **知识需求**: `knowledge`
* **推理类型**: `type`, `reasoning_type`

### 示例数据行 (Example Data)

```json
{"id": "q_001", "question": "What color is the car?", "answer": "red", "prediction": "the car is red", "category": "color", "hop": "1"}
{"id": "q_002", "question": "List the triples.", "answer": "[['Bob', 'works_at', 'Apple']]", "prediction": "Bob | works_at | Apple", "category": "graph understand triple", "hop": "2"}
```

## 📊 评估指标说明 (Metrics)

1.  **VQA Accuracy (常规问题)**：
    * 将生成的答案和标准答案全部转为小写。
    * 去除标点符号、冠词、并将文字数字转为阿拉伯数字（如 "one" -> "1"）。
    * 如果标准答案的词语被完全包含在预测句子中，则计为正确（Correct = 1）。
2.  **Triple Exact Match (图理解问题)**：
    * 当 `category` 为 `graph understand triple` 时触发。
    * 将字符串或列表格式的预测值解析为三元组集合。
    * 预测的三元组集合与真实三元组集合必须长度一致，且元素一一对应匹配（支持无序），才计为正确。
3.  **List Answer (列表类问题)**：
    * 当 `category` 为 `color_list` 或 `text` 时触发。
    * 要求预测的列表元素与真实列表元素完全匹配（基于 VQA 宽松匹配逻辑）。

## 📝 产出文件说明

### 1. 结果数据文件 (`*_evaluated.jsonl`)
在原始文件的基础上，每一行新增了一个布尔/整型字段 `"correct"`：
```json
// 原始行
{"id": 1, "answer": "yes", "prediction": "yes"}
// 评测后
{"id": 1, "answer": "yes", "prediction": "yes", "correct": 1}
```

### 2. 统计报告 (`*_summary.txt`)
生成的 txt 文件会直观地展示各维度的准确率：
```text
==== Overall ====
samples = 1000
accuracy = 85.50%

==== Accuracy by Category ====
category                  count   acc
color                     200     90.00%
graph understand          300     82.33%
hop reasoning(1)          250     88.00%
hop reasoning(2/3)        250     81.20%

==== Accuracy by Hop ====
...
```

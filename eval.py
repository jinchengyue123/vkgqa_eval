# -*- coding: utf-8 -*-
import argparse
import ast
import json
import re
from typing import Any, List, Optional, Tuple

import pandas as pd
pd.options.display.encoding = 'utf-8'

# -------------------- VQAEval (未修改) --------------------
def normalize_text(s):
    """小写化、去标点、去掉多余空格"""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def has_word(sentence, word):
    """
    宽松匹配，用于 VQA / direction / paraphrase 类任务
    """
    s = normalize_text(sentence)
    w = normalize_text(word)

    s_tokens = set(s.split())
    w_tokens = set(w.split())

    # 常见停用词，防止 "and" / "or" 影响
    stopwords = {"and", "or", "the", "a", "an", "of", "to"}
    s_tokens -= stopwords
    w_tokens -= stopwords

    # 要求每个 word token 都至少在 sentence 出现一次
    return all(token in s_tokens for token in w_tokens)


class VQAEval:
    def __init__(self):
        self.contractions = {
            "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
            "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
            "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
            "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
            "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
            "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
            "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've", "isnt": "isn't",
            "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll",
            "let's": "let's", "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've", "mightve": "might've", "mustnt": "mustn't",
            "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock",
            "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at",
            "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's",
            "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", "somebodys": "somebody's",
            "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've",
            "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
            "somethingd've": "something'd've", "something'dve": "something'd've", "somethingll": "something'll",
            "thats": "that's", "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've",
            "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
            "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
            "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've",
            "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're",
            "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",
            "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
            "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've",
            "whyll": "why'll", "whyre": "why're", "whys": "why's", "wont": "won't",
            "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
            "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
            "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll",
            "youre": "you're", "youve": "you've",
        }
        self.manualMap = {
            "none": "0","zero": "0","one": "1","two": "2","three": "3","four":"4",
            "five":"5","six":"6","seven":"7","eight":"8","nine":"9","ten":"10",
        }
        self.articles = ["a","an","the"]
        self.periodStrip = re.compile(r"(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile(r"(\d)(\,)(\d)")
        self.punct = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_",
                      "-", ">", "<", "@", "`", ",", "?", "!"]

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (re.search(self.commaStrip, inText) != None):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = " ".join(outText)
        return outText

    def evaluate(self, answer, gt_answers):
        answer = answer.replace("\n"," ").replace("\t"," ").strip()
        answer = self.processPunctuation(answer)
        answer = self.processDigitArticle(answer)
        if isinstance(gt_answers, list):
            for i in range(len(gt_answers)):
                g = gt_answers[i].replace("\n"," ").replace("\t"," ").strip()
                g = self.processPunctuation(g)
                g = self.processDigitArticle(g)
                if has_word(answer, g):
                    return 1
            return 0
        else:
            g = gt_answers.replace("\n"," ").replace("\t"," ").strip()
            g = self.processPunctuation(g)
            g = self.processDigitArticle(g)
            return 1 if has_word(answer, g) else 0

# -------------------- Triple 解析 (未修改) --------------------
Triple = Tuple[str, str, str]

def _to_list(obj: Any) -> Optional[List]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, (tuple, dict)):
        return [obj]
    if isinstance(obj, str):
        s = obj.strip()
        for loader in (json.loads, ast.literal_eval):
            try:
                v = loader(s)
                if isinstance(v, (list, tuple, dict)):
                    return v if isinstance(v, list) else [v]
            except Exception:
                pass
        if s:
            parts = [x.strip() for x in re.split(r"[;\n]+", s) if x.strip()]
            return parts if parts else None
    return None

def _triple_from_one(o: Any) -> Optional[Triple]:
    if isinstance(o, (list, tuple)) and len(o) == 3:
        h, r, t = map(str, o); return h, r, t
    if isinstance(o, dict):
        low = {k.lower(): v for k, v in o.items()}
        for hk, rk, tk in [("head","relation","tail"),("h","r","t"),("subject","predicate","object")]:
            if hk in low and rk in low and tk in low:
                return str(low[hk]), str(low[rk]), str(low[tk])
    if isinstance(o, str):
        for sep in [r"\|", ","]:
            parts = [x.strip() for x in re.split(sep, o) if x.strip()]
            if len(parts) == 3:
                return parts[0], parts[1], parts[2]
    return None

def parse_triples(field: Any) -> List[Triple]:
    triples: List[Triple] = []
    lst = _to_list(field)
    if lst is None:
        return []
    if len(lst) == 1 and isinstance(lst[0], str):
        tr = _triple_from_one(lst[0])
        return [tr] if tr else []
    for item in lst:
        tr = _triple_from_one(item)
        if tr:
            triples.append(tr)
        elif isinstance(item, str):
            for seg in [x.strip() for x in re.split(r"[;\n]+", item) if x.strip()]:
                tr2 = _triple_from_one(seg)
                if tr2: triples.append(tr2)
    return triples

# -------------------- 评估函数 (未修改) --------------------
def _triple_match_vqa(pred_triple, gt_triple, vqa) -> bool:
    hp, rp, tp = pred_triple
    hg, rg, tg = gt_triple
    return (
        vqa.evaluate(hp, hg) == 1 and
        vqa.evaluate(rp, rg) == 1 and
        vqa.evaluate(tp, tg) == 1
    )

def eval_triple_exact_em(pred: str, gt, vqa) -> int:
    gt_list = parse_triples(gt)
    pred_list = parse_triples(pred)
    if not gt_list or not pred_list: return 0
    if len(gt_list) != len(pred_list): return 0
    used = [False] * len(gt_list)
    for p in pred_list:
        found = False
        for j, g in enumerate(gt_list):
            if not used[j] and _triple_match_vqa(p, g, vqa):
                used[j] = True
                found = True
                break
        if not found:
            return 0
    return 1

def eval_list_answer(pred, gt, vqa) -> int:
    pred_list = _to_list(pred)
    gt_list = _to_list(gt)
    if pred_list is None or gt_list is None:
        return 0
    if len(pred_list) != len(gt_list):
        return 0
    used = [False] * len(pred_list)
    for g in gt_list:
        found = False
        for i, p in enumerate(pred_list):
            if not used[i] and vqa.evaluate(p, g) == 1:
                used[i] = True
                found = True
                break
        if not found:
            return 0
    return 1

def eval_non_triple(pred: str, gt, category: str, vqa) -> int:
    if category.lower() in ["color_list", "text"]:
        return eval_list_answer(pred, gt, vqa)
    else:
        return vqa.evaluate(pred, gt)

# -------------------- Main (核心修改部分) --------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate a single JSONL file containing both ground truth and predictions.")
    ap.add_argument("--input", required=True, help="Input JSONL file containing both predictions and ground truth.")
    ap.add_argument("--out-jsonl", help="Output JSONL file (defaults to input_name_evaluated.jsonl).")
    ap.add_argument("--out-txt", help="Output summary TXT file (defaults to input_name_summary.txt).")
    args = ap.parse_args()

    in_name = args.input.rsplit(".", 1)[0]
    out_jsonl = args.out_jsonl if args.out_jsonl else f"{in_name}_evaluated.jsonl"
    out_txt = args.out_txt if args.out_txt else f"{in_name}_summary.txt"

    # 读取原始 JSONL 数据（使用内置 json 库保证输出时结构和类型完美保留）
    records = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    if not records:
        print("Input file is empty.")
        return

    # 动态匹配列名（寻找 gt 和 pred 的 key）
    def _find_key(keys, candidates):
        cmap = {k.lower(): k for k in keys}
        for c in candidates:
            if c.lower() in cmap:
                return cmap[c.lower()]
        return None

    first_keys = list(records[0].keys())
    # 严格区分 gt_key 和 pred_key 的候选项，防止匹配冲突
    gt_key = _find_key(first_keys, ["gt_answers", "label", "ground_truth", "answer", "answers"])
    pred_key = _find_key(first_keys, ["prediction", "pred", "output", "model_answer", "response"])
    cat_key = _find_key(first_keys, ["category", "cat", "type", "task_type"])
    hop_key = _find_key(first_keys, ["hop", "hops", "num_hop", "num_hops"])
    know_key = _find_key(first_keys, ["knowledge"])
    type_key = _find_key(first_keys, ["type", "reasoning_type", "reason_type", "reasoning_cat"])

    if not gt_key or not pred_key:
        raise ValueError(f"Could not automatically detect ground truth column or prediction column. Found keys: {first_keys}")

    vqa = VQAEval()
    recs_for_summary = []

    # 逐行评估
    for row in records:
        gt = str(row.get(gt_key, ""))
        pred = str(row.get(pred_key, ""))
        cat = str(row.get(cat_key, "")) if cat_key and row.get(cat_key) is not None else ""
        hop = str(row.get(hop_key, "")) if hop_key and row.get(hop_key) is not None else ""
        knowledge_raw = row.get(know_key) if know_key else None
        type_val = str(row.get(type_key, "")) if type_key and row.get(type_key) is not None else ""

        # 执行评测
        is_triple = (cat.strip().lower() == "graph understand triple")
        if is_triple:
            ok = eval_triple_exact_em(pred, gt, vqa)
        else:
            ok = eval_non_triple(pred, gt, cat, vqa)

        # ✨ 核心：在原数据上直接新增 correct 属性
        row["correct"] = ok

        # 清洗 knowledge 用于统计（保持你原先的代码逻辑）
        if knowledge_raw is None or str(knowledge_raw).lower() == "null":
            knowledge = "null"
        else:
            knowledge = str(int(float(knowledge_raw))) if str(knowledge_raw).replace(".","",1).isdigit() else str(knowledge_raw)

        recs_for_summary.append({
            "category": cat,
            "hop": hop,
            "knowledge": knowledge,
            "type": type_val,
            "correct": ok
        })

    # 将增加了 'correct' 的完整字典重新写入 JSONL
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[OK] Evaluated samples saved to -> {out_jsonl}")

    # ==============================================================
    # 以下为 Pandas 统计逻辑（完美复刻你原来的聚合方式）
    # ==============================================================
    df_sample = pd.DataFrame(recs_for_summary)

    category_map = {
        "graph understand triple": "graph understand",
        "graph understand": "graph understand",
        "color_list": "color",
        "color": "color"
    }
    df_sample["category"] = df_sample["category"].astype(str).str.strip().str.lower().map(lambda x: category_map.get(x, x))

    total_samples = len(df_sample)
    overall_acc = df_sample["correct"].mean() if total_samples > 0 else 0.0

    def _split_hop_reasoning(row):
        c = str(row["category"]).strip().lower()
        h = str(row["hop"]).strip()
        if c == "hop reasoning":
            if h == "1": return "hop reasoning(1)"
            elif h in ["2", "3"]: return "hop reasoning(2/3)"
        return c

    df_sample["category_split"] = df_sample.apply(_split_hop_reasoning, axis=1)

    by_cat = df_sample.groupby("category_split", dropna=False)["correct"].agg(["count", "mean"]).reset_index().rename(columns={"mean": "acc", "category_split": "category"})
    by_hop = df_sample.groupby("hop", dropna=False)["correct"].agg(["count", "mean"]).reset_index().rename(columns={"mean": "acc"})
    by_knowledge = df_sample.groupby("knowledge", dropna=False)["correct"].agg(["count", "mean"]).reset_index().rename(columns={"mean": "acc"})
    
    # Reasoning type 统计
    if type_key:
        by_reasoning_type = (
            df_sample[df_sample["category"].str.contains("reasoning", case=False, na=False)]
            .groupby("type", dropna=False)["correct"]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={"mean": "acc"})
        )
    else:
        by_reasoning_type = pd.DataFrame(columns=["type", "count", "acc"])

    def _fmt_percent(x):
        return f"{x*100:.2f}%"

    def _build_report_text(total, overall_acc, by_cat, by_hop, by_knowledge):
        def pct_df(df: pd.DataFrame, key_name: str):
            if df is None or df.empty:
                return pd.DataFrame(columns=[key_name, "count", "acc"])
            out = df.copy()
            out["acc"] = out["acc"].apply(lambda v: _fmt_percent(float(v)) if pd.notna(v) else "")
            return out
        lines = []
        lines.append("==== Overall ====")
        lines.append(f"samples = {total}")
        lines.append(f"accuracy = {_fmt_percent(overall_acc)}\n")
        lines.append("==== Accuracy by Category ====")
        lines.append(pct_df(by_cat, "category").to_string(index=False))
        lines.append("\n==== Accuracy by Hop ====")
        lines.append(pct_df(by_hop, "hop").to_string(index=False))
        lines.append("\n==== Accuracy by Knowledge ====")
        lines.append(pct_df(by_knowledge, "knowledge").to_string(index=False))
        lines.append("\n==== Accuracy by Reasoning Type ====")
        lines.append(pct_df(by_reasoning_type, "type").to_string(index=False))
        return "\n".join(lines)

    report_text = _build_report_text(total_samples, overall_acc, by_cat, by_hop, by_knowledge)
    
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[OK] summary (text) -> {out_txt}")
    print("\n" + report_text)

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
import argparse
import ast
import json
import re
from typing import Any, List, Optional, Tuple

import pandas as pd
pd.options.display.encoding = 'utf-8'

# -------------------- VQAEval --------------------
import re

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

# -------------------- Triple 解析 --------------------
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

# -------------------- Triple 评估 --------------------
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

# -------------------- List 答案评估 --------------------
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

# -------------------- 非 triple 答案评估 --------------------
def eval_non_triple(pred: str, gt, category: str, vqa) -> int:
    if category.lower() in ["color_list", "text"]:
        return eval_list_answer(pred, gt, vqa)
    else:
        return vqa.evaluate(pred, gt)

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--per-sample-out", required=True)
    ap.add_argument("--summary-out", required=True)
    args = ap.parse_args()

    # 读取数据
    df_gt = pd.read_json(args.tsv, lines=True, dtype=str)
    df_pred = pd.read_json(args.pred, lines=True, dtype=str)

    # 确保 knowledge 列统一
    df_pred.rename(columns={'knowledge': 'knowledge_column'}, inplace=True)
    df_gt.rename(columns={'knowledge': 'knowledge_column'}, inplace=True)

    df_ref = df_gt

    # 自动检测列
    def _find_col(df, candidates):
        cmap = {c.lower(): c for c in df.columns}
        for c in candidates:
            if c.lower() in cmap:
                return cmap[c.lower()]
        return None

    id_gt = _find_col(df_gt, ["index", "id"]) or "index"
    id_pred = _find_col(df_pred, ["index", "id"]) or "index"
    gt_col = _find_col(df_gt, ["answer", "answers", "gt_answers", "label"])
    pred_col = _find_col(df_pred, ["prediction", "answer", "pred", "output", "model_answer", "response"])
    cat_col_src = _find_col(df_pred, ["category", "cat", "type", "task_type"]) \
                  or _find_col(df_gt, ["category", "cat", "type", "task_type"])
    hop_col_src = _find_col(df_pred, ["hop", "hops", "num_hop", "num_hops"]) \
                  or _find_col(df_gt, ["hop", "hops", "num_hop", "num_hops"])

    # 合并数据
    df_gt[id_gt] = df_gt[id_gt].astype(str)
    df_pred[id_pred] = df_pred[id_pred].astype(str)
    keep_cols = [id_pred, pred_col] + ([cat_col_src] if cat_col_src else []) + ([hop_col_src] if hop_col_src else []) + ["knowledge_column"]
    merged = pd.merge(
        df_gt[[id_gt, gt_col]],
        df_pred[keep_cols],
        left_on=id_gt, right_on=id_pred, how="inner",
        suffixes=("_tsv", "_pred")
    )

    gt_col_m = gt_col
    pred_col_m = pred_col
    cat_col_m = cat_col_src
    hop_col_m = hop_col_src

    # per-sample 评估
    vqa = VQAEval()
    recs = []
    for _, r in merged.iterrows():
        idx = r[id_gt]
        gt = r[gt_col_m]
        pred = r[pred_col_m]
        cat = str(r[cat_col_m]) if cat_col_m and pd.notna(r[cat_col_m]) else ""
        hop = str(r[hop_col_m]) if hop_col_m and pd.notna(r[hop_col_m]) else ""

        is_triple = (cat.strip().lower() == "graph understand triple")
        if is_triple:
            ok = eval_triple_exact_em(pred, gt, vqa)
        else:
            ok = eval_non_triple(pred, gt, cat, vqa)

        knowledge = r.get("knowledge_column", None)
        if pd.isna(knowledge) or knowledge == "null":
            knowledge = "null"
        else:
            knowledge = str(int(float(knowledge))) if str(knowledge).replace(".","",1).isdigit() else str(knowledge)

        recs.append({
            "index": idx, "answer": gt, "prediction": pred,
            "category": cat, "hop": hop, "knowledge": knowledge, "correct": ok
        })

    df_sample = pd.DataFrame(recs)

    # Category 归并
    category_map = {
        "graph understand triple": "graph understand",
        "graph understand": "graph understand",
        "color_list": "color",
        "color": "color"
    }

    df_sample["category"] = (
        df_sample["category"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda x: category_map.get(x, x))
    )

    # Accuracy 统计
    total_samples = len(df_sample)
    overall_acc = df_sample["correct"].mean() if total_samples > 0 else 0.0

    # 新增：根据 hop 字段，把 hop reasoning 类别细分
    def _split_hop_reasoning(row):
        cat = str(row["category"]).strip().lower()
        hop = str(row["hop"]).strip()
        if cat == "hop reasoning":
            if hop == "1":
                return "hop reasoning(1)"
            elif hop in ["2", "3"]:
                return "hop reasoning(2/3)"
        return cat

    df_sample["category_split"] = df_sample.apply(_split_hop_reasoning, axis=1)

    # 分组统计
    by_cat = (
        df_sample.groupby("category_split", dropna=False)["correct"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"mean": "acc", "category_split": "category"})
    )
    by_hop = (
        df_sample.groupby("hop", dropna=False)["correct"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"mean": "acc"})
    )
    by_knowledge = (
        df_sample.groupby("knowledge", dropna=False)["correct"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"mean": "acc"})
    )
    # ==== 按类别统计 ====
    by_category = (
        df_sample.groupby("category", dropna=False)["correct"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"mean": "acc"})
    )

    # ==== 按 reasoning type 统计 ====
    type_col = _find_col(df_ref, ["type", "reasoning_type", "reason_type", "reasoning_cat"])
    if type_col and type_col in df_ref.columns:
        merged_type = merged.copy()
        merged_type["type"] = df_ref[type_col].astype(str).str.strip().str.lower()
        df_sample["type"] = merged_type["type"]
        by_reasoning_type = (
            df_sample[df_sample["category"].str.contains("reasoning", case=False, na=False)]
            .groupby("type", dropna=False)["correct"]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={"mean": "acc"})
        )
    else:
        by_reasoning_type = pd.DataFrame(columns=["type", "count", "acc"])

    # 输出 per-sample
    df_sample.to_json(args.per_sample_out, orient="records", lines=True, force_ascii=False)
    print(f"[OK] per-sample -> {args.per_sample_out}")

    # 输出 summary text
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
    with open(args.summary_out, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[OK] summary (text) -> {args.summary_out}")
    print("\n" + report_text)

    # summary JSONL
    jsonl_out = args.summary_out.rsplit(".",1)[0]+".jsonl"
    summary_dict = {
        "total_samples": total_samples,
        "overall_acc": overall_acc,
        "by_category": by_cat.to_dict(orient="records"),
        "by_hop": by_hop.to_dict(orient="records"),
        "by_knowledge": by_knowledge.to_dict(orient="records"),
        "by_reasoning_type": by_reasoning_type.to_dict(orient="records")
    }
    with open(jsonl_out, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, ensure_ascii=False, indent=2)
    print(f"[OK] summary (jsonl) -> {jsonl_out}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_questions(path: Path) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for row in load_jsonl(path):
        out[str(row["question_id"])] = row
    return out


def load_predictions(path: Path) -> Tuple[Dict[str, str], Dict[str, Dict]]:
    pred: Dict[str, str] = {}
    meta: Dict[str, Dict] = {}
    for row in load_jsonl(path):
        qid = str(row["question_id"])
        pred[qid] = str(row.get("text", "")).strip()
        if "visiontrim" in row:
            meta[qid] = row["visiontrim"]
    return pred, meta


def build_report(
    questions: Dict[str, Dict],
    vanilla: Dict[str, str],
    visiontrim: Dict[str, str],
    visiontrim_meta: Dict[str, Dict],
    top_k: int,
) -> Dict:
    common_ids = sorted(set(vanilla) & set(visiontrim))
    changed_ids = [qid for qid in common_ids if vanilla[qid] != visiontrim[qid]]

    yn = {"yes", "no"}
    yn_flip = 0
    yn_to_open = 0
    open_to_yn = 0
    for qid in changed_ids:
        v_is_yn = vanilla[qid] in yn
        t_is_yn = visiontrim[qid] in yn
        if v_is_yn and t_is_yn:
            yn_flip += 1
        elif v_is_yn and not t_is_yn:
            yn_to_open += 1
        elif (not v_is_yn) and t_is_yn:
            open_to_yn += 1

    cat_total = Counter()
    cat_changed = Counter()
    for qid in common_ids:
        category = questions.get(qid, {}).get("category", "unknown")
        cat_total[category] += 1
        if qid in changed_ids:
            cat_changed[category] += 1

    suspicious_patterns = {
        "contains_digit": re.compile(r"\d"),
    }
    suspicious_counts = {
        "vanilla": Counter(),
        "visiontrim": Counter(),
    }

    def mark_suspicious(method: str, ans: str) -> None:
        if len(ans.split()) >= 3:
            suspicious_counts[method][">=3_words"] += 1
        if len(ans) > 25:
            suspicious_counts[method]["len>25"] += 1
        for key, pat in suspicious_patterns.items():
            if pat.search(ans):
                suspicious_counts[method][key] += 1
        if ans.startswith("[error]"):
            suspicious_counts[method]["error"] += 1

    for ans in vanilla.values():
        mark_suspicious("vanilla", ans)
    for ans in visiontrim.values():
        mark_suspicious("visiontrim", ans)

    pair_counter = Counter((vanilla[qid], visiontrim[qid]) for qid in changed_ids)
    top_changed_pairs = [
        {"vanilla": a, "visiontrim": b, "count": c}
        for (a, b), c in pair_counter.most_common(top_k)
    ]

    changed_examples = []
    for qid in changed_ids[:top_k]:
        q = questions.get(qid, {})
        changed_examples.append(
            {
                "question_id": qid,
                "category": q.get("category", "unknown"),
                "question": str(q.get("text", "")).split("\n")[0],
                "vanilla": vanilla[qid],
                "visiontrim": visiontrim[qid],
            }
        )

    meta_summary = {}
    if visiontrim_meta:
        keys = [
            "orig_img_tokens",
            "kept_img_tokens",
            "dvts_tokens",
            "tgvc_tokens",
            "orig_seq_len",
            "new_seq_len",
        ]
        n = len(visiontrim_meta)
        avg = {k: 0.0 for k in keys}
        for m in visiontrim_meta.values():
            for k in keys:
                avg[k] += float(m.get(k, 0))
        for k in keys:
            avg[k] /= n
        keep_ratio = avg["kept_img_tokens"] / avg["orig_img_tokens"] if avg["orig_img_tokens"] else 0.0
        seq_ratio = avg["new_seq_len"] / avg["orig_seq_len"] if avg["orig_seq_len"] else 0.0
        meta_summary = {
            "samples": n,
            "avg": avg,
            "keep_ratio": keep_ratio,
            "sequence_ratio": seq_ratio,
        }

    category_breakdown = []
    for cat in sorted(cat_total.keys()):
        total = cat_total[cat]
        changed = cat_changed[cat]
        category_breakdown.append(
            {
                "category": cat,
                "total": total,
                "changed": changed,
                "changed_rate": (changed / total) if total else 0.0,
            }
        )

    return {
        "total_common": len(common_ids),
        "vanilla_only": len(set(vanilla) - set(visiontrim)),
        "visiontrim_only": len(set(visiontrim) - set(vanilla)),
        "same": len(common_ids) - len(changed_ids),
        "changed": len(changed_ids),
        "agreement": ((len(common_ids) - len(changed_ids)) / len(common_ids)) if common_ids else 0.0,
        "yes_no_transition": {
            "yes_no_flip": yn_flip,
            "yes_no_to_open": yn_to_open,
            "open_to_yes_no": open_to_yn,
        },
        "suspicious_counts": {
            "vanilla": dict(suspicious_counts["vanilla"]),
            "visiontrim": dict(suspicious_counts["visiontrim"]),
        },
        "visiontrim_meta": meta_summary,
        "category_breakdown": category_breakdown,
        "top_changed_pairs": top_changed_pairs,
        "changed_examples": changed_examples,
    }


def render_markdown(report: Dict) -> str:
    lines: List[str] = []
    lines.append("# GQA Prediction Comparison")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- common samples: {report['total_common']}")
    lines.append(f"- same answers: {report['same']}")
    lines.append(f"- changed answers: {report['changed']}")
    lines.append(f"- agreement: {report['agreement']:.4f}")
    lines.append(f"- vanilla-only ids: {report['vanilla_only']}")
    lines.append(f"- visiontrim-only ids: {report['visiontrim_only']}")
    lines.append("")

    yn = report["yes_no_transition"]
    lines.append("## Yes/No Transitions")
    lines.append(f"- yes<->no flip: {yn['yes_no_flip']}")
    lines.append(f"- yes/no -> open: {yn['yes_no_to_open']}")
    lines.append(f"- open -> yes/no: {yn['open_to_yes_no']}")
    lines.append("")

    lines.append("## Suspicious Counts")
    for method in ("vanilla", "visiontrim"):
        lines.append(f"- {method}: {report['suspicious_counts'][method]}")
    lines.append("")

    meta = report.get("visiontrim_meta", {})
    if meta:
        avg = meta["avg"]
        lines.append("## VisionTrim Meta")
        lines.append(f"- samples: {meta['samples']}")
        lines.append(f"- avg orig_img_tokens: {avg['orig_img_tokens']:.2f}")
        lines.append(f"- avg kept_img_tokens: {avg['kept_img_tokens']:.2f}")
        lines.append(f"- keep_ratio: {meta['keep_ratio']:.4f}")
        lines.append(f"- avg orig_seq_len: {avg['orig_seq_len']:.2f}")
        lines.append(f"- avg new_seq_len: {avg['new_seq_len']:.2f}")
        lines.append(f"- sequence_ratio: {meta['sequence_ratio']:.4f}")
        lines.append("")

    lines.append("## Category Breakdown")
    for row in report["category_breakdown"]:
        lines.append(
            f"- {row['category']}: total={row['total']}, changed={row['changed']}, changed_rate={row['changed_rate']:.4f}"
        )
    lines.append("")

    lines.append("## Top Changed Pairs")
    for row in report["top_changed_pairs"]:
        lines.append(f"- {row['vanilla']} -> {row['visiontrim']}: {row['count']}")
    lines.append("")

    lines.append("## Changed Examples")
    for row in report["changed_examples"]:
        lines.append(
            f"- {row['question_id']} | {row['category']} | Q: {row['question']} | vanilla: {row['vanilla']} | visiontrim: {row['visiontrim']}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two GQA prediction jsonl files.")
    parser.add_argument("--question-file", required=True, type=Path)
    parser.add_argument("--vanilla-file", required=True, type=Path)
    parser.add_argument("--visiontrim-file", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    questions = load_questions(args.question_file)
    vanilla, _ = load_predictions(args.vanilla_file)
    visiontrim, vt_meta = load_predictions(args.visiontrim_file)

    report = build_report(
        questions=questions,
        vanilla=vanilla,
        visiontrim=visiontrim,
        visiontrim_meta=vt_meta,
        top_k=args.top_k,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with args.out_md.open("w", encoding="utf-8") as f:
        f.write(render_markdown(report))

    print(f"[Done] JSON report: {args.out_json}")
    print(f"[Done] Markdown report: {args.out_md}")


if __name__ == "__main__":
    main()

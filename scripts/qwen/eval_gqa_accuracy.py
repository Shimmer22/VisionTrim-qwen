#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict


def normalize(text: str) -> str:
    return text.strip().rstrip(".").lower()


def load_gold(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(qid): normalize(str(item["answer"])) for qid, item in data.items()}


def load_pred_jsonl(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[str(row["question_id"])] = normalize(str(row.get("text", "")))
    return out


def evaluate(gold: Dict[str, str], pred: Dict[str, str]) -> Dict:
    common_ids = sorted(set(gold) & set(pred))
    correct = sum(1 for qid in common_ids if pred[qid] == gold[qid])
    full = len(gold)
    covered = len(common_ids)
    return {
        "gold_total": full,
        "pred_total": len(pred),
        "coverage_count": covered,
        "coverage_rate": (covered / full) if full else 0.0,
        "correct": correct,
        "accuracy_on_covered": (correct / covered) if covered else 0.0,
        "accuracy_over_full_gold": (correct / full) if full else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GQA prediction jsonl against testdev_balanced_questions.json")
    parser.add_argument("--gold-questions", required=True, type=Path)
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--name", default="run", type=str)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    gold = load_gold(args.gold_questions)
    pred = load_pred_jsonl(args.pred)
    result = {"name": args.name, **evaluate(gold, pred)}

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[Done] wrote {args.out_json}")


if __name__ == "__main__":
    main()

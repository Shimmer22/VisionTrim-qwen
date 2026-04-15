#!/usr/bin/env python3
import argparse
import json
import random
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description='Create <=N subset for GQA jsonl and images')
    p.add_argument('--input-jsonl', required=True)
    p.add_argument('--input-images', required=True, help='directory with all images')
    p.add_argument('--output-jsonl', required=True)
    p.add_argument('--output-images', required=True)
    p.add_argument('--max-samples', type=int, default=10000)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def resolve_image_name(item):
    for key in ('image', 'img', 'image_path', 'image_file'):
        if key in item and item[key]:
            return str(item[key]).split('/')[-1]
    if 'image_id' in item and item['image_id']:
        iid = str(item['image_id'])
        if iid.endswith('.jpg') or iid.endswith('.png'):
            return iid
        return iid + '.jpg'
    return None


def main():
    args = parse_args()
    in_jsonl = Path(args.input_jsonl)
    in_images = Path(args.input_images)
    out_jsonl = Path(args.output_jsonl)
    out_images = Path(args.output_images)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    rows = []
    with in_jsonl.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    total = len(rows)
    keep = min(args.max_samples, total)
    rng = random.Random(args.seed)
    idx = list(range(total))
    rng.shuffle(idx)
    keep_idx = set(idx[:keep])

    selected = [rows[i] for i in range(total) if i in keep_idx]

    copied = 0
    missing = 0
    seen = set()

    with out_jsonl.open('w', encoding='utf-8') as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            img_name = resolve_image_name(item)
            if not img_name or img_name in seen:
                continue
            seen.add(img_name)
            src = in_images / img_name
            if not src.exists() and img_name.endswith('.jpg'):
                alt = in_images / (img_name[:-4] + '.png')
                if alt.exists():
                    src = alt
            if src.exists():
                shutil.copy2(src, out_images / src.name)
                copied += 1
            else:
                missing += 1

    print(json.dumps({
        'total_questions': total,
        'selected_questions': keep,
        'unique_images_copied': copied,
        'missing_images': missing,
        'output_jsonl': str(out_jsonl),
        'output_images': str(out_images),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

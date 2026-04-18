#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GQA inference with Qwen2.5-VL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--method", type=str, default="vanilla", choices=["vanilla", "visiontrim"])
    parser.add_argument("--retain-ratio", type=float, default=0.33, help="Retained visual token ratio for visiontrim")
    parser.add_argument("--dvts-ratio", type=float, default=0.75, help="Fraction of retained tokens allocated to DVTS")
    parser.add_argument("--tgvc-iter", type=int, default=1, help="TGVC refinement iterations")
    parser.add_argument("--use-mps", action="store_true", help="Force use MPS if available")
    return parser.parse_args()


def load_questions(path: str) -> List[Dict]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Question file not found: {path}")

    if file_path.suffix == ".jsonl":
        rows = []
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "questions" in data and isinstance(data["questions"], list):
            return data["questions"]
        return list(data.values())
    raise ValueError(f"Unsupported question file format: {path}")


def chunk_items(items: List[Dict], chunk_idx: int, num_chunks: int) -> List[Dict]:
    if num_chunks <= 1:
        return items
    return [item for i, item in enumerate(items) if i % num_chunks == chunk_idx]


def pick_device(force_mps: bool = False) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if (force_mps or True) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def resolve_image_path(image_folder: str, item: Dict) -> str:
    image_folder_path = Path(image_folder)
    candidates: List[Path] = []

    for key in ("image", "img", "image_path", "image_file"):
        if key in item:
            value = str(item[key])
            path = Path(value)
            if path.is_absolute():
                candidates.append(path)
            else:
                candidates.append(image_folder_path / value)

    if "image_id" in item:
        image_id = str(item["image_id"])
        candidates.extend(
            [
                image_folder_path / image_id,
                image_folder_path / f"{image_id}.jpg",
                image_folder_path / f"{image_id}.png",
            ]
        )

    for cand in candidates:
        if cand.exists():
            return str(cand)

    raise FileNotFoundError(
        f"Cannot resolve image for question id {item.get('question_id', item.get('questionId', 'N/A'))}: {candidates[:3]}"
    )


def get_question_text(item: Dict) -> str:
    for key in ("text", "question", "query"):
        if key in item and item[key]:
            return str(item[key])
    raise KeyError(f"No question text key found in item: {item.keys()}")


def get_question_id(item: Dict) -> str:
    for key in ("question_id", "questionId", "qid", "id"):
        if key in item:
            return str(item[key])
    raise KeyError(f"No question id key found in item: {item.keys()}")


def normalize_answer(text: str) -> str:
    return text.strip().rstrip(".").lower()


def _split_text_image_positions(input_ids_1d: torch.Tensor, image_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    image_pos = (input_ids_1d == image_token_id).nonzero(as_tuple=False).flatten()
    text_pos = (input_ids_1d != image_token_id).nonzero(as_tuple=False).flatten()
    return text_pos, image_pos


def _local_affinity_scores(image_embeds: torch.Tensor, window: int = 2) -> torch.Tensor:
    # 1D neighborhood proxy for LTAM when flattened vision tokens are used.
    n = image_embeds.size(0)
    if n <= 1:
        return torch.ones(n, device=image_embeds.device, dtype=image_embeds.dtype)
    feat = torch.nn.functional.normalize(image_embeds, dim=-1)
    scores = torch.zeros(n, device=image_embeds.device, dtype=image_embeds.dtype)
    for i in range(n):
        left = max(0, i - window)
        right = min(n, i + window + 1)
        neigh = feat[left:right]
        sim = torch.matmul(neigh, feat[i])
        scores[i] = sim.mean()
    scores = torch.softmax(scores, dim=0)
    return scores

# 采取方差来判断全局和局部重要性哪个更可靠，从而自适应地融合两者的排序分数，得到最终的token重要性排序。
def _variance_adaptive_fusion(global_scores: torch.Tensor, local_scores: torch.Tensor) -> torch.Tensor:
    var_g = torch.var(global_scores, unbiased=False)
    var_l = torch.var(local_scores, unbiased=False)
    alpha = var_l / (var_g + var_l + 1e-8) 
    # alpha大，说明local scores更分散，应该更信任global scores；
    return alpha * global_scores + (1.0 - alpha) * local_scores


def _tgvc_merge(
    remain_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    r_tokens: int,
    iters: int = 1,
) -> torch.Tensor:
    # 输入:
    # remain_embeds: 经过DVTS后剩余的视觉token, 形状 [Nr, D]
    # text_embeds: 文本token embedding, 形状 [Lt, D]
    # r_tokens: 目标压缩后要得到的TGVC token数量
    # iters: 细化迭代次数
    # 输出:
    # centers: 压缩后的视觉token, 形状 [r_tokens, D]
    if r_tokens <= 0 or remain_embeds.size(0) == 0:
        # 没有可压缩token或目标数量非法时, 返回空张量
        return remain_embeds.new_zeros((0, remain_embeds.size(-1)))

    # 目标数量不能超过当前剩余token数量
    r_tokens = min(r_tokens, remain_embeds.size(0))
    # 归一化文本特征, 后续点积可视作余弦相似度
    text_norm = torch.nn.functional.normalize(text_embeds, dim=-1)
    remain = remain_embeds

    # 第1步: 选择初始中心（Center selection）
    # 依据文本到视觉的相关性, 选出最相关的r_tokens个视觉token作为初始中心
    remain_norm = torch.nn.functional.normalize(remain, dim=-1)
    # t2v: 每个文本token对每个视觉token的相关性分布 [Lt, Nr]
    t2v = torch.softmax(torch.matmul(text_norm, remain_norm.T), dim=-1)
    # 对所有文本token求平均, 得到每个视觉token的综合分数 [Nr]
    token_score = t2v.mean(dim=0)
    # 取分数最高的r_tokens个视觉token作为初始中心
    center_idx = torch.topk(token_score, k=r_tokens).indices
    centers = remain[center_idx]

    # 第2步: 迭代细化中心（Refinement）
    # 思路: 先把每个视觉token分配给最匹配的中心, 再做加权聚合更新中心
    for _ in range(max(1, iters)):
        if remain.size(0) <= r_tokens:
            # token数量已不大于目标数量, 无需继续压缩
            break
        # 归一化中心, 便于与文本做相似度计算
        center_norm = torch.nn.functional.normalize(centers, dim=-1)
        # v2t: 每个视觉token对文本的相关性分布 [Nr, Lt]
        v2t = torch.softmax(torch.matmul(remain_norm, text_norm.T), dim=-1)  # [Nr, Lt]
        # c2t: 每个中心对文本的相关性分布 [R, Lt]
        c2t = torch.softmax(torch.matmul(center_norm, text_norm.T), dim=-1)  # [R, Lt]
        # assign_score[i, j]: 第i个视觉token与第j个中心在“文本语义分布”上的匹配度
        assign_score = torch.matmul(v2t, c2t.T)  # [Nr, R]
        # 每个视觉token分配到匹配度最高的中心
        assign = torch.argmax(assign_score, dim=-1)

        merged = []
        for j in range(r_tokens):
            # 找到分配给第j个中心的所有视觉token下标
            idx = (assign == j).nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                # 若该中心没有分配到token, 保持原中心不变
                merged.append(centers[j])
                continue
            # 仅使用该簇内token对中心j的匹配度做归一化权重
            weights = assign_score[idx, j]
            weights = weights / (weights.sum() + 1e-8)
            # 对簇内token做加权聚合, 得到该簇的残差信息
            agg = (remain[idx] * weights.unsqueeze(-1)).sum(dim=0)
            # 更新中心: 原中心 + 聚合残差
            merged.append(centers[j] + agg)
        # 堆叠为新的中心集合 [R, D]
        centers = torch.stack(merged, dim=0)
        # remain固定不变, 这里只需要更新其归一化结果供下一轮使用
        remain_norm = torch.nn.functional.normalize(remain, dim=-1)

    # 返回压缩后的TGVC token
    return centers


def _visiontrim_compress(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    retain_ratio: float,
    dvts_ratio: float,
    tgvc_iter: int,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    n = image_embeds.size(0)
    if n == 0:
        return image_embeds, {"n": 0, "k": 0, "r": 0, "m": 0}

    m = max(1, int(round(n * retain_ratio)))
    m = min(m, n)
    k = max(1, int(round(m * dvts_ratio)))
    k = min(k, m)
    r = m - k

    # DVTS proxy:
    # 首先做DVTS的token选择，选出k个token作为dominant tokens，剩余的作为TGVC的输入。
    # 对token对embedding求范数：以此作为重要性排序（全局）
    global_scores = torch.softmax(torch.norm(image_embeds, dim=-1), dim=0)
    # 局部重要性取决于与邻近token的相似度：以此作为重要性排序（局部）
    local_scores = _local_affinity_scores(image_embeds)
    # 全局和局部重要性自适应融合，得到最终的token重要性排序。
    fused = _variance_adaptive_fusion(global_scores, local_scores)
    topk_idx = torch.topk(fused, k=k).indices
    v_dom = image_embeds[topk_idx]

    if r <= 0:
        return v_dom, {"n": n, "k": k, "r": 0, "m": m}

    keep_mask = torch.ones(n, dtype=torch.bool, device=image_embeds.device)
    keep_mask[topk_idx] = False
    remain = image_embeds[keep_mask]

    # TGVC proxy:
    # 对于剩下的视觉token，基于文本-视觉相关性进行聚类合并，得到r个TGVC tokens。
    v_com = _tgvc_merge(remain, text_embeds, r_tokens=r, iters=tgvc_iter)
    v_final = torch.cat([v_dom, v_com], dim=0)
    return v_final, {"n": n, "k": k, "r": r, "m": m}

# prepare to compact vision tokens with DVTS and TGVC before feeding into the model for generation.
def _prepare_visiontrim_inputs(model, processor, device: str, image_path: str, question: str, retain_ratio: float, dvts_ratio: float, tgvc_iter: int):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {
                    "type": "text",
                    "text": f"Question: {question}\nAnswer with a short phrase only.",
                },
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    input_ids = inputs["input_ids"]  # [1, L]
    attention_mask = inputs["attention_mask"]  # [1, L]
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")

    token_embeds = model.get_input_embeddings()(input_ids)  # [1, L, D]
    # 获得：经过视觉编码器，完成了patchmerger之后的特征（即将和文本拼接）
    image_features = model.get_image_features(pixel_values, image_grid_thw).pooler_output # get image features
    # 将多段段图片特征拼接成一个整体
    image_embeds = torch.cat(image_features, dim=0).to(token_embeds.device, token_embeds.dtype)  # [Nv, D]

    text_pos, image_pos = _split_text_image_positions(input_ids[0], model.config.image_token_id)
    if image_pos.numel() != image_embeds.size(0):
        raise ValueError(
            f"Image placeholder/token mismatch: image_pos={image_pos.numel()} vs image_embeds={image_embeds.size(0)}"
        )
    text_embeds = token_embeds[0, text_pos, :]

    compressed_image_embeds, stats = _visiontrim_compress(
        image_embeds=image_embeds,
        text_embeds=text_embeds,
        retain_ratio=retain_ratio,
        dvts_ratio=dvts_ratio,
        tgvc_iter=tgvc_iter,
    )

    first_img = int(image_pos.min().item())
    last_img = int(image_pos.max().item())
    left_ids = input_ids[0, :first_img]
    right_ids = input_ids[0, last_img + 1 :]
    m = compressed_image_embeds.size(0)
    new_image_ids = torch.full((m,), model.config.image_token_id, dtype=input_ids.dtype, device=input_ids.device)
    new_input_ids_1d = torch.cat([left_ids, new_image_ids, right_ids], dim=0)

    left_emb = token_embeds[0, :first_img, :]
    right_emb = token_embeds[0, last_img + 1 :, :]
    new_embeds_2d = torch.cat([left_emb, compressed_image_embeds, right_emb], dim=0)

    new_input_ids = new_input_ids_1d.unsqueeze(0)
    new_inputs_embeds = new_embeds_2d.unsqueeze(0)
    new_attention_mask = torch.ones_like(new_input_ids, dtype=attention_mask.dtype, device=attention_mask.device)

    meta = {
        "orig_img_tokens": int(stats["n"]),
        "kept_img_tokens": int(stats["m"]),
        "dvts_tokens": int(stats["k"]),
        "tgvc_tokens": int(stats["r"]),
        "orig_seq_len": int(input_ids.size(1)),
        "new_seq_len": int(new_input_ids.size(1)),
    }
    return new_input_ids, new_inputs_embeds, new_attention_mask, meta


def infer_one(
    model,
    processor,
    device: str,
    image_path: str,
    question: str,
    max_new_tokens: int,
    temperature: float,
    method: str,
    retain_ratio: float,
    dvts_ratio: float,
    tgvc_iter: int,
):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {
                    "type": "text",
                    "text": f"Question: {question}\nAnswer with a short phrase only.",
                },
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    do_sample = temperature > 0
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature

    trim_meta = None
    with torch.inference_mode():
        if method == "vanilla":
            generated_ids = model.generate(**inputs, **generate_kwargs)
            prompt_len = inputs["input_ids"].shape[1]
        else:
            # here comes to the visiontrim path with dynamic token selection and TGVC compression.
            new_input_ids, new_inputs_embeds, new_attention_mask, trim_meta = _prepare_visiontrim_inputs(
                model=model.model,
                processor=processor,
                device=device,
                image_path=image_path,
                question=question,
                retain_ratio=retain_ratio,
                dvts_ratio=dvts_ratio,
                tgvc_iter=tgvc_iter,
            )
            generated_ids = model.generate(
                input_ids=new_input_ids,
                inputs_embeds=new_inputs_embeds,
                attention_mask=new_attention_mask,
                **generate_kwargs,
            )
            prompt_len = new_input_ids.shape[1]

    generated_ids_trimmed = [out_ids[prompt_len:] for out_ids in generated_ids]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return output_text, trim_meta


def main() -> None:
    args = parse_args()

    questions = load_questions(args.question_file)
    questions = chunk_items(questions, args.chunk_idx, args.num_chunks)
    os.makedirs(Path(args.answers_file).parent, exist_ok=True)

    device = pick_device(force_mps=args.use_mps)
    torch_dtype = get_dtype(device)

    print(
        f"[Info] device={device}, dtype={torch_dtype}, model={args.model}, method={args.method}, "
        f"retain_ratio={args.retain_ratio}"
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=None,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    with open(args.answers_file, "w", encoding="utf-8") as fout:
        printed_trim_stats = False
        for item in tqdm(questions, desc="GQA inference"):
            qid = get_question_id(item)
            question = get_question_text(item)
            image_path = resolve_image_path(args.image_folder, item)
            try:
                answer, trim_meta = infer_one(
                    model=model,
                    processor=processor,
                    device=device,
                    image_path=image_path,
                    question=question,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    method=args.method,
                    retain_ratio=args.retain_ratio,
                    dvts_ratio=args.dvts_ratio,
                    tgvc_iter=args.tgvc_iter,
                )
                if args.method == "visiontrim" and trim_meta is not None and not printed_trim_stats:
                    print(f"[VisionTrim] sample stats: {trim_meta}")
                    printed_trim_stats = True
            except Exception as exc:
                answer = f"[ERROR] {exc}"
                trim_meta = None

            out = {
                "question_id": qid,
                "text": normalize_answer(answer),
                "model_id": args.model,
                "method": args.method,
            }
            if trim_meta is not None:
                out["visiontrim"] = trim_meta
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[Done] Wrote predictions to {args.answers_file}")


if __name__ == "__main__":
    main()

# VisionTrim 论文主体方法总结（聚焦复现）

- 论文: VisionTrim: Unified Vision Token Compression for Training-Free MLLM Acceleration
- 版本: arXiv v3（2026-03-28），ICLR 2026
- 目标: 训练无关（training-free）地压缩视觉 token，在尽量不掉点的前提下加速 MLLM 推理。

## 1. 方法动机

VisionTrim 的核心判断是：
- 现有方法大多只在一个阶段做 token 压缩（要么视觉编码、要么 LLM 解码）。
- 单阶段压缩容易带来跨模态对齐退化，尤其是文本相关的细粒度视觉信息被误删。

因此论文提出统一框架，覆盖“视觉编码 + LLM 解码”两阶段，并用文本引导补全机制减少误删影响。

## 2. 总体框架

VisionTrim 由两个可插拔模块组成：
- DVTS: Dominant Vision Token Selection
- TGVC: Text-Guided Vision Complement

给定视觉 token `V in R^{Nxd}`:
1. DVTS 先选出主导 token `V_dom in R^{Kxd}`。
2. TGVC 对剩余 token 做文本引导聚合，得到补全 token `V_com in R^{Rxd}`。
3. 拼接成最终输入 `V_final=[V_dom;V_com] in R^{(K+R)xd}`，其中 `K+R < N`。

## 3. DVTS: 主导视觉 token 选择

DVTS 用“全局语义 + 局部连续性”双评分，再做自适应融合。

### 3.1 全局语义分数（Global Semantic Importance）
- 使用视觉编码器倒数第二层 `[CLS]` 对各视觉 token 的注意力。
- 对多头注意力取平均得到 `S_g`，再归一化得到全局重要性。
- 直觉: 与全局语义最相关的 token 应该保留。

### 3.2 局部连续性分数（LTAM）
- 在局部 `k x k` 邻域里计算 token 亲和度。
- 亲和度由两部分组成:
  - 特征相似度核（feature kernel）
  - 空间位置邻近核（position kernel）
- 得到局部重要性 `S_l`。
- 直觉: 视觉结构连续区域不应被破坏。

### 3.3 自适应融合与 Top-K
- 融合公式: `S_i = alpha * S_g_hat + (1-alpha) * S_l`。
- `alpha` 由全局/局部分数方差自适应确定，方差越稳定的分支权重越高。
- 按 `S_i` 取 Top-K，得到 `V_dom`。

## 4. TGVC: 文本引导视觉补全

TGVC 不直接丢弃剩余 token，而是“压缩成少量文本相关补全 token”。

输入:
- 剩余视觉 token `V_r in R^{(N-K)xd}`
- 文本特征 `T in R^{Lxd}`

步骤:
1. 计算 `text-to-vision` 相似度，选 Top-R 视觉 token 作为聚类中心。
2. 其余 token 依据文本引导相似度分配到各中心。
3. 在每个簇内按相似度加权聚合，生成 `R` 个补全 token `V_com`。
4. 经过迭代细化后，与 `V_dom` 拼接成 `V_final`。

关键收益:
- 避免“只剪不补”导致的文本相关证据丢失。
- 提升跨模态对齐，降低幻觉和知识边界漂移风险。

## 5. 两阶段集成策略（Multi-stage Pruning）

VisionTrim 可在两个位置启用：

1. 视觉编码阶段:
- 在视觉 token 进入 LLM 之前压缩，直接减少后续计算负担。

2. LLM 解码阶段:
- 可插入任意两层 Transformer 间进行动态压缩。
- 该阶段不再用 `[CLS]`，而是利用“首个生成 token”对视觉 token 的注意力作为全局语义信号。
- 结合视觉-文本跨模态注意力，继续执行 DVTS + TGVC。

论文默认报告里常在浅层插入（例如第 2-3 层之间）获得更稳性能/效率平衡。

## 6. 与已有方法的主要差异

相比典型训练无关压缩方法（如只做 pruning 的路线），VisionTrim 的差异是：
- 从单阶段升级为全链路压缩（ViT 编码 + LLM 解码）。
- 从单一评分升级为全局语义 + 局部连续性的联合建模。
- 从“纯剪枝”升级为“主导保留 + 文本引导补全”的两步策略。
- 强调跨模态对齐，不把文本注意力当唯一裁剪依据。

## 7. 与你当前复现目标的关系（Qwen2.5-VL）

- 论文主表展示了 Qwen2-VL-7B 与 Qwen2.5-VL-7B 的迁移结果，说明 VisionTrim 在 Qwen 家族上是可迁移的。
- 你计划使用 `Qwen/Qwen2.5-VL-3B-Instruct`，属于同家族小模型，方法上可参考同样的两阶段思路。
- 但当前仓库缺少完整 `llava/` 核心实现目录，不能直接按仓库脚本一键重跑，需要你后续补齐实现或单独写 Qwen 路线代码。

## 8. 当前环境状态（与你的 Apple 加速要求相关）

我已创建环境 `visiontrim-qwen25vl` 并安装了 `torch/torchvision/transformers/accelerate/qwen-vl-utils`。

验证结果:
- `torch.__version__ = 2.11.0`
- `torch.backends.mps.is_built() = True`
- `torch.backends.mps.is_available() = False`

解释:
- `is_built=True` 说明安装的是支持 Apple MPS 的 PyTorch 构建。
- `is_available=False` 说明当前运行上下文没有检测到可用 MPS 设备（常见于受限会话、系统权限或运行环境差异）。

建议你在自己的交互终端里再次验证:

```bash
conda activate visiontrim-qwen25vl
python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_built(), torch.backends.mps.is_available())"
```

若需先跑通流程可开启回退:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## 9. 主要公式定位（便于你二次核对）

- DVTS 全局语义: Eq.(1)-(3)
- LTAM 局部连续性: Eq.(4)-(5)
- 自适应融合: Eq.(6)
- TGVC 中心选择/分配/聚合: Eq.(7)-(11)
- 解码阶段跨模态打分: Eq.(12)

## 10. 参考链接

- arXiv abs: https://arxiv.org/abs/2601.22674
- arXiv pdf: https://arxiv.org/pdf/2601.22674
- 项目仓库: https://github.com/hanxunyu/VisionTrim

# FisherSteer

本仓库按 `论文初稿_中文.md` 第 3 节与附录 A.3 实现 FisherSteer。保留原 PRI 的脚本名、命令行参数和输出目录，便于继续使用已有实验入口。

## 代码结构

- `exp/train_pri_detector.py`：读取数据、提取最后 m 个输入词元的跨层状态、训练三分类 MLP、校准 Fisher 统计及选层。
- `utils/fisher.py`：类别等权协方差、正则化线性求解、Fisher 得分与 Top-K 层选择。
- `utils/hidden_states.py`：统一读取各 decoder block 的原始输出（最终 model norm 之前）。
- `exp/evaluate_asr.py`：检测、条件二次预填充、逐层当前状态干预、KV 缓存续写及 ASR 评估。
- `exp/utility_overrefusal.py`：MT-Bench、XSTest、OR-Bench；生成部分复用 ASR 的相同实现。
- `exp/batch_asr.py`：五模型、八攻击批量入口。
- `data/`：原始训练与评估数据；`models/`：本地模型；`training_results/`：训练产物；`results/`：评估结果。

## 方法与实现

MLP 对每个词元的 L×d 状态逐层做 L2 归一化，再执行 LayerNorm → Linear(d,256) → ReLU → Dropout(0.2) → Linear(256,128) → ReLU → 层均值池化 → Linear(128,64) → ReLU → Dropout(0.2) → Linear(64,3)。默认 AdamW、80 epochs、学习率 1e-3、weight decay 1e-4、batch size 16、seed 42；保存验证准确率最高的检查点。

Fisher 统计使用**未归一化**的隐藏状态，分别在每个相对词元位置、每一层计算：

```text
Sigma_c = mean_i((h_i-mu_c)(h_i-mu_c)^T)
Sigma_W = (Sigma_J + Sigma_B + Sigma_H) / 3
tau = trace(Sigma_W) / d
C = Sigma_W + rho * tau * I
v_J = mu_H - mu_J, v_B = mu_H - mu_B
w_c = solve(C, v_c), R_Hc = v_c^T w_c
S_l = mean_t(R_HJ[t,l] + R_HB[t,l])
```

使用完整协方差；样本数小于维度时用等价 Woodbury 求解，避免保存所有 d×d 矩阵。产物保存均值、v、w、R、tau、rho 和格式版本。按当前稿件第 3.2 节，rho 默认取 **0.1**；可通过 `--fisher-rho` 显式指定。仅当 tau 小于数值下限时使用 `max(tau, fisher_variance_floor)`，默认下限 1e-12。

首次预填充检测最后 m 个词元。默认只要有一个 J 即触发，仅修改 J 位置；未触发直接复用首次预填充缓存。触发时重新预填充，在每个选定层读取已经受前面层影响的**当前**状态 h，并执行：

```text
lambda = w^T(mu_H - h) / (v^T w)
h' = h + lambda * v
```

lambda 保留正负，不裁剪。v=0 时不修改。完成后删除干预 hooks，使用选定预填充的 KV cache 正常逐词元解码。`--pri-j-consecutive-trigger 2`、`--pri-j-only false` 仍可用于消融，均偏离论文默认设置。

## 数据处理与兼容性

原始 JSON 保持不变。读取后按类别和完整提示精确去重，同一提示跨 train/val 出现时优先保留 train；之后再检查两侧完整输入不重叠。`data_preparation.json` 记录去重前后各类别/划分数量。实际数量可能与论文中的描述不同，需要作者核对，代码不会生成缺失的主题匹配良性数据或虚构实验结果。

MLP 与类别统计均只使用训练划分；验证划分用于检查点选择。没有声明 split 时按完整输入记录 ID 划分；所有词元共享输入归属。Base64/Zulu 继续排除。训练与推理共用模型提示构建逻辑，包括优化攻击的模板约定。

**旧 PRI 检测器、余弦选层文件与欧氏投影校准产物不兼容，必须重新训练和校准。** 加载旧产物会明确报错；不会静默混用。模型层号为 1-based，t=1 为最后一个输入词元。

## 安装与运行

安装适合硬件的 PyTorch，然后执行 `pip install -r requirements.txt`。`models/` 支持 `vicuna-7b-v1.5`、`vicuna-13b-v1.5`、`Llama-2-7b-chat-hf`、`Mistral-7B-Instruct-v0.2`、`Meta-Llama-3-8B-Instruct`；也可传 `--model-path`。

```bash
python exp/train_pri_detector.py --model-name vicuna-7b --prefill-num-tokens 10 --td-topk-m-select 10 --td-topk-max-k 10 --fisher-rho 0.1
python exp/evaluate_asr.py --model-name vicuna-7b --attack gcg --data-split test --num-samples 50 --td-topk-k 10 --disable-GPT-judge
python exp/batch_asr.py --disable-GPT-judge
python exp/utility_overrefusal.py --model-name vicuna-7b --dataset xstest --disable-GPT-judge
python -m unittest discover -s tests -v
```

Fisher 校准默认在 CPU 以 float64 逐位置、逐层求解；可通过 `--fisher-device cuda:0` 指定设备，应确保显存充足。每模型需要独立训练和校准；完整高维协方差计算比旧余弦评分耗时。

默认测试划分遵循原仓库留出策略，Template 使用其完整 30 条测试样本。禁用 GPT 时只运行本地拒答字符串判断，其结果不能等同论文 GPT-4o 五分制 ASR。需要外部评分时传 `--GPT_API`（默认评判模型已设为 `gpt-4o`，可用 `--judge-model` 指定）（会调用外部 API）。

## 验证范围

`tests/test_fishersteer.py` 包含直接协方差求解对照（含类别数量不平衡与 Woodbury 分支）、退化情形、投影约束、多层当前状态干预、hook 清理、数据划分检查和随机初始化微型 Llama 的缓存续写测试。无需下载模型或调用 API。完整模型的 ASR、FPR、效用和时间开销需在权重与计算资源齐备后重新评估。

复查补充：训练特征保存为 float32，避免 CPU float32/BF16 输出被额外量化到 float16；选层和检测必须使用同一 m，加载时检查选层产物的 rho 与干预校准 rho 一致。

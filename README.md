# FisherSteer

Code repository for the paper **FisherSteer: Fisher-Calibrated Activation
Steering for LLM Jailbreak Defense**.

FisherSteer is a white-box jailbreak defense that detects jailbreak-related
hidden states during the prefill phase. It uses regularized Fisher scores to
select intervention layers and adaptively steers the detected states toward
the mean projection of directly harmful requests. Autoregressive decoding then
proceeds without further detection or intervention.

## Repository layout

```text
FisherSteer-main/
├── data/                  # Training and evaluation datasets
├── exp/                   # Training and evaluation entry points
├── models/                # Local model weights (not tracked by Git)
├── training_results/      # Detector and Fisher calibration artifacts
├── results/               # Evaluation outputs
├── mt_bench/              # MT-Bench resources
├── peft/                  # Bundled PEFT package
├── tests/                 # Numerical and integration tests
└── utils/                 # Fisher, hidden-state, model, and prompt utilities
```

## Model directories

Place the target models under `models/` using the following directory names:

```text
models/
├── vicuna-7b-v1.5/
├── vicuna-13b-v1.5/
├── Llama-2-7b-chat-hf/
├── Mistral-7B-Instruct-v0.2/
└── Meta-Llama-3-8B-Instruct/
```

The scripts also accept an explicit local path through `--model-path`.

## Installation

Create a Python environment with the CUDA-compatible PyTorch build required by
your server, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Train and calibrate FisherSteer

The training entry point reads the corresponding JSON file from `data/train/`
unless `--dataset-path` is supplied. Base64 and Zulu samples are excluded.
Exact duplicate prompts are removed before feature extraction, with training
samples taking precedence over validation duplicates.

```bash
python exp/train_pri_detector.py \
  --model-name vicuna-7b \
  --prefill-num-tokens 10 \
  --td-topk-m-select 10 \
  --td-topk-max-k 10 \
  --fisher-rho 0.1
```

Critical layers are ranked using regularized Fisher scores computed from class
means and the equally weighted within-class covariance of jailbreak, benign,
and directly harmful requests. The regularization strength defaults to
`rho = 0.1`. Layer selection and detection use the same number of trailing
input tokens.

Train and calibrate a separate detector for each target model. Artifacts are
saved under `training_results/`. The evaluation scripts automatically select
the latest training run for the requested model; use `--artifact-dir` to select
a specific run.

## ASR evaluation

Run one model/attack setting:

```bash
python exp/evaluate_asr.py \
  --model-name vicuna-7b \
  --attack gcg \
  --data-split test \
  --num-samples 50 \
  --pri-j-only true \
  --pri-j-consecutive-trigger 1 \
  --GPT_API YOUR_API_KEY
```

By default, any trailing input token classified as jailbreak triggers a second
prefill pass. Only tokens classified as jailbreak are modified at the selected
layers. If no jailbreak token is detected, generation continues from the first
prefill pass.

Run the complete five-model, eight-attack grid:

```bash
python exp/batch_asr.py --GPT_API YOUR_API_KEY
```

ASR outputs are written to `results/ASR/`. GPT-based judging uses `gpt-4o` by
default; override it with `--judge-model`. Pass the API key explicitly with
`--GPT_API YOUR_API_KEY`, or replace that option with `--disable-GPT-judge` to
use local refusal-string evaluation only.

The default `test` split excludes the train/validation prefix recorded by the
dataset construction policy. `Template` therefore evaluates its complete
30-sample held-out split even when `--num-samples 50` is used.

## Utility and over-refusal evaluation

```bash
python exp/utility_overrefusal.py \
  --model-name vicuna-7b \
  --dataset xstest
```

Supported datasets are `mt_bench`, `xstest`, and `or-bench-hard-1k`. Outputs
are written to `results/utility_overrefusal/`. MT-Bench judging requires
`--GPT_API YOUR_API_KEY`; use `--disable-GPT-judge` to run without GPT-based
judging. If FastChat cannot construct a conversation template, the evaluation
code automatically falls back to the built-in model-specific template.

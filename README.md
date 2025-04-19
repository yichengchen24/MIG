# MIG: Automatic Data Selection for Instruction Tuning by Maximizing Information Gain in Semantic Space

🤗[HF Models](https://huggingface.co/collections/xsample/mig-models-6801ec964bab5e098a676f19) 🤗[HF Datasets](https://huggingface.co/collections/xsample/mig-datasets-6800b4d225243877293eff3b) 📄[Paper]() 🚀[Project](https://yichengchen24.github.io/projects/mig/)

Welcome to MIG (**M**aximize the **I**nformation **G**ain) Project!

We will continue to update, please stay tuned!

## What is MIG?
MIG is an automatic data selection method for instruction tuning. It proposes an information-based dataset measurement that comprehensively evaluates data quality and diversity.

## 🔥 News
* 📄 [04/2025] MIG paper [MIG: Automatic Data Selection for Instruction Tuning by Maximizing Information Gain in Semantic Space]() is released!

## Performance

### 🔦 Highlights
<img src="./assets/teaser-v6.png" alt="x" width="400">

Comparison with different data selection methods:
* Sample 50K from the Tulu3 pool(939K).
* Training on Llama3.1-8B.
* Comprehensive evaluations including human-preference and knowledge-based benchmarks.

### 📈 Full Results

| Method  | Data Size | ARC          | BBH          | GSM          | HE           | MMLU         | IFEval       | $Avg_\text{obj}$ | AE           | MT          | Wild          | $Avg_\text{sub}$ | Avg          |
| ------- | --------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ---------------- | ------------ | ----------- | ------------- | ---------------- | ------------ |
| Pool    | 939K      | 69.15        | 63.88        | 83.40        | 63.41        | 65.77        | 67.10        | 68.79            | 8.94         | 6.86        | -24.66        | 38.40            | 53.59        |
| Random  | 50K       | 74.24        | 64.80        | 70.36        | 51.22        | 63.86        | 61.00        | 64.25            | 8.57         | <u>7.06</u> | -22.15        | 39.36            | 51.81        |
| ZIP     | 50K       | 77.63        | 63.00        | 52.54        | 35.98        | 65.00        | 61.00        | 59.19            | 6.71         | 6.64        | -32.10        | 35.69            | 47.44        |
| IFD     | 50K       | 75.93        | 63.56        | 61.03        | 49.39        | 64.39        | 53.60        | 61.32            | 12.30        | 7.03        | -20.20        | 40.83            | 51.08        |
| #InsTag | 50K       | 72.54        | 64.80        | 69.83        | 48.17        | 63.50        | **65.99**    | 64.14            | 6.58         | 6.84        | -20.70        | 38.21            | 51.17        |
| DEITA   | 50K       | 78.98        | 66.11        | **74.07**    | 49.39        | 64.00        | 64.33        | <u>66.15</u>     | 10.19        | 6.83        | <u>-19.95</u> | 39.50            | 52.83        |
| CaR     | 50K       | 78.98        | **69.04**    | 71.42        | 52.44        | **65.15**    | 56.75        | 65.63            | 12.55        | 6.95        | -20.67        | 40.57            | 53.10        |
| QDIT    | 50K       | <u>79.66</u> | 65.42        | 70.74        | <u>53.05</u> | <u>65.06</u> | 57.30        | 65.21            | **15.78**    | 6.76        | -20.56        | <u>41.03</u>     | <u>53.12</u> |
| MIG     | 50K       | **80.00**    | <u>66.39</u> | <u>72.02</u> | **57.93**    | 64.44        | <u>65.06</u> | **67.64**        | <u>14.66</u> | **7.32**    | **-17.77**    | **42.99**        | **55.32**    |

HE denotes HumanEval, AE denotes AlpacaEvalv2, MT denotes MTBench, and Wild denotes WildBench. $Avg_\text{obj}$ and $Avg_\text{sub}$ represent the average of the normalized knowledge-based and human-preference benchmark scores, respectively. Avg is the mean of $Avg_\text{obj}$ and $Avg_\text{sub}$.

Please refer to our paper for more results on different data pools(Openhermes2.5, Deita-Sota-Pool) and different base LLMs(Qwen2.5-7B, Mistral-7B-v0.3).

## 🏃‍♂️ How to start?

### Installation
* Create an environment
```shell
conda create -n xsample python=3.10
conda activate xsample
```
* Install pytorch (>2.0)
```shell
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```
* Install
```shell
git clone https://github.com/yichengchen24/xsample.git
cd xsample
pip install -e .
```
* Embedding Model

Please download embedding model under <embedding_model_path>, we recommand [e5-mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct) used in our paper.

### Data Sampling
```bash
xsample sample <src> --out <save_path> --num-sample <num_sample> --valid-tag-path ./configs/valid_tag_path.json --label-graph-type sim --embedding-model <embedding_model_path> --sampler-type mig --batch-size 32768
```

<src> should be the data pool path in format of jsonl. Please refer to `data/example.jsonl` for an example. We have open-sourced our processed data pools([Tulu3](https://huggingface.co/datasets/xsample/tulu-3-pool-annotated), [Openhermes2.5](https://huggingface.co/datasets/xsample/openhermes-2.5-pool-annotated), [$X_{sota}$](https://huggingface.co/datasets/xsample/deita-sota-pool-annotated)) with annotated [#InsTag](https://github.com/OFA-Sys/InsTag) labels and [Deita](https://github.com/hkust-nlp/deita) score.

### SFT Training
We use [LLama-Factory](https://github.com/hiyouga/LLaMA-Factory) to fine-tuning base models.

* Add sampled data in `data/dataset_info.json`

```json
"tulu3_pool_mig_50k": {
    "file_name": <out>,
    "formatting": "sharegpt",
    "columns": {
      "messages": "dialogs"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
}
```

* Training
```bash
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} src/train.py \
    --stage sft \
    --model_name_or_path meta-llama/Llama-3.1-8B \
    --do_train \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --dataset tulu3_pool_mig_50k \
    --cutoff_len 4096 \
    --template llama3 \
    --finetuning_type full \
    --output_dir  ckpts/tulu3_pool_mig_50k \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type linear \
    --logging_steps 100 \
    --save_steps 5000 \
    --learning_rate 5e-6 \
    --num_train_epochs 3.0 \
    --warmup_ratio 0.03 \
    --plot_loss \
    --bf16 \
    --save_only_model
```


## 💪 What's more?

We will continue to update including:

- [ ] More automatic data selection strategies


## Citation
If you find the content of this project helpful, please cite our paper as follows:

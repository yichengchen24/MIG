import os.path as osp
from copy import deepcopy

from mmengine.config import read_base
from opencompass.models import (
    HuggingFacewithChatTemplate,
    TurboMindModelwithChatTemplate,
)

from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

#######################################################################
#                          PART 0  Essential Configs                  #
#######################################################################
with read_base():
    # Datasets Part
    from opencompass.configs.datasets.ARC_c.ARC_c_cot_gen_926652 import (
        ARC_c_datasets,
    )
    from opencompass.configs.datasets.bbh.bbh_gen_5b92b0 import bbh_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_a58960 import \
        gsm8k_datasets
    from opencompass.configs.datasets.humaneval.humaneval_gen_8e312c import \
        humaneval_datasets
    from opencompass.configs.datasets.mmlu.mmlu_gen_4d595a import mmlu_datasets
    from opencompass.configs.datasets.IFEval.IFEval_gen_3321a3 import \
        ifeval_datasets
        
    # Summarizer
    from opencompass.configs.summarizers.groups.bbh import bbh_summary_groups
    from opencompass.configs.summarizers.groups.mmlu import mmlu_summary_groups
    

    # Model List
    # from opencompass.configs.models.qwen.lmdeploy_qwen2_1_5b_instruct import models as lmdeploy_qwen2_1_5b_instruct_model
    # from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat import models as hf_internlm2_5_7b_chat_model
    # from opencompass.configs.models.openbmb.hf_minicpm_2b_sft_bf16 import models as hf_minicpm_2b_sft_bf16_model
    # from opencompass.configs.models.yi.hf_yi_1_5_6b_chat import models as hf_yi_1_5_6b_chat_model
    # from opencompass.configs.models.gemma.hf_gemma_2b_it import models as hf_gemma_2b_it_model
    # from opencompass.configs.models.yi.hf_yi_1_5_34b_chat import models as hf_yi_1_5_34b_chat_model

#######################################################################
#                          PART 1  Datasets List                      #
#######################################################################
# datasets list for evaluation
datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

#######################################################################
#                       PART 2  Datset Summarizer                     #
#######################################################################
# with read_base():

core_summary_groups = [
    {
        'name':
        'core_average',
        'subsets': [
            ['ARC-c', 'accuracy'],
            ['bbh', 'naive_average'],
            ['gsm8k', 'accuracy'],
            ['openai_humaneval', 'humaneval_pass@1'],
            ['mmlu', 'naive_average'],
            ['IFEval', 'Prompt-level-strict-accuracy'],
        ],
    },
]

summarizer = dict(
    dataset_abbrs=[
        ['core_average', 'naive_average'],
        ['ARC-c', 'accuracy'],
        ['bbh', 'naive_average'],
        ['gsm8k', 'accuracy'],
        ['openai_humaneval', 'humaneval_pass@1'],
        ['mmlu', 'naive_average'],
        ['IFEval', 'Prompt-level-strict-accuracy'],
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)

#######################################################################
#                        PART 3  Models  List                         #
#######################################################################

models = []
model_configs = [
    ('Llama-3.1-8B-Instruct', '/cpfs01/user/chenyicheng/aliyun/ckpts/meta-llama/Llama-3.1-8B-Instruct', 1),
]

max_seq_len = 16384
max_out_len = 4096
max_batch_size = 128

for abbr, path, num_gpus in model_configs:
    if abbr is None:
        abbr = path.split('/')[-2] + '--' + path.split('/')[-1]

    base_model = dict(
        type=TurboMindModelwithChatTemplate,
        abbr=abbr,
        path=path,
        engine_config=dict(session_len=max_seq_len,
                           max_batch_size=max_batch_size,
                           tp=num_gpus),
        gen_config=dict(top_k=1,
                        temperature=1e-6,
                        top_p=0.9,
                        max_new_tokens=max_out_len),
        max_seq_len=max_seq_len,
        max_out_len=max_out_len,
        batch_size=max_batch_size,
        run_cfg=dict(num_gpus=num_gpus),
    )

    hf_base_model = dict(
        type=HuggingFacewithChatTemplate,
        abbr=abbr,
        path=path,
        max_out_len=max_out_len,
        batch_size=8,
        run_cfg=dict(num_gpus=num_gpus),
    )

    model = deepcopy(base_model)
    if 'TurboMindModelwithChatTemplate' in str(model['type']):
        model['gen_config']['top_k'] = 1  # greedy
        model['gen_config']['temperature'] = 1e-6
        models.append(model)
    else:
        models.append(model)

#######################################################################
#                 PART 4  Inference/Evaluation Configuaration         #
#######################################################################

# Local Runner
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        retry=0,  # Modify if needed
        task=dict(type=OpenICLInferTask)),
)

# eval with local runner
eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(type=LocalRunner,
                max_num_workers=16,
                task=dict(type=OpenICLEvalTask)),
)

#######################################################################
#                      PART 5  Utils Configuaration                   #
#######################################################################
base_exp_dir = 'outputs/corebench_v1_9/'
work_dir = osp.join(base_exp_dir, 'chat_objective')

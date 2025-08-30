from copy import deepcopy
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.subjective.alpaca_eval.alpacav2_judgeby_gpt4 import alpacav2_datasets
    from opencompass.configs.datasets.subjective.wildbench.wildbench_pair_judge import wildbench_datasets
    from opencompass.configs.datasets.subjective.multiround.mtbench_single_judge_diff_temp import mtbench_datasets

from opencompass.models import (
    HuggingFacewithChatTemplate,
    TurboMindModelwithChatTemplate,
    OpenAI,
)
from opencompass.partitioners import NaivePartitioner, SizePartitioner, NumWorkerPartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.partitioners.sub_num_worker import \
    SubjectiveNumWorkerPartitioner
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.runners import LocalRunner, SlurmSequentialRunner
from opencompass.summarizers import SubjectiveSummarizer
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])

#######################################################################
#                          PART 1  Datasets List                      #
#######################################################################
datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

#######################################################################
#                       PART 2  Dataset Summarizer                    #
#######################################################################
summarizer = dict(type=SubjectiveSummarizer, function='subjective')

#######################################################################
#                        PART 3  Models  List                         #
#######################################################################

models = []
sampling_models = []
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
        engine_config=dict(
            session_len=max_seq_len, max_batch_size=max_batch_size, tp=num_gpus
        ),
        gen_config=dict(
            top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=max_out_len
        ),
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

    model = deepcopy(base_model)
    if 'TurboMindModelwithChatTemplate' in str(model['type']):
        model['gen_config']['top_k'] = 1000
        model['gen_config']['temperature'] = 1
        sampling_models.append(model)
    else:
        model['generation_kwargs'] = dict(do_sample=True)
        sampling_models.append(model)

#######################################################################
#                 PART 4  Inference/Evaluation Configuaration         #
#######################################################################
# judge_models = [
#     dict(
#         abbr='GPT4-Turbo',
#         type=OpenAI,
#         path='gpt-4-1106-preview',
#         key=
#         'xxxx',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
#         meta_template=api_meta_template,
#         query_per_second=16,
#         max_out_len=2048,
#         max_seq_len=2048,
#         batch_size=8,
#         temperature=0,
#     )
# ]
judge_models = [
    dict(
        abbr='CompassJudger-1-32B-Instruct',
        type=OpenAI,
        openai_api_base='http://22.8.10.237:23333:10001/v1/chat/completions',  # Change to your own url if needed.
        path='opencompass/CompassJudger-1-32B-Instruct',
        key='YOUR_API_KEY',
        rpm_verbose=True,
        meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=2048,
        max_seq_len=4096,
        temperature=0.01,
        batch_size=64,
        retry=20,
        tokenizer_path='gpt-4o-2024-05-13',
    )
]

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
    partitioner=dict(
        type=SubjectiveNaivePartitioner,
        models=models,
        judge_models=judge_models,
    ),
    runner=dict(type=LocalRunner,
                max_num_workers=16,
                task=dict(type=SubjectiveEvalTask)),
)

#######################################################################
#                      PART 5  Utils Configuaration                   #
#######################################################################
work_dir = 'outputs/subjective/'

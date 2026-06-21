# LoftNN: Distributed Training of Deep Learning Models in the Edge-Cloud-Space Continuum

LoftNN is a framework for distributed training in the edge-cloud-space continuum.

All details on LoftNN can be found in the paper (more information will be provided here after the publication).

## Installation and development setup

LoftNN has been implemented using Python 3.10.

A setup script is provided to automate the creation of a virtual environment and the installation of LoftNN and its dependencies. Use this script when a simple, CPU-only setup is sufficient:
```
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

Alternatively, you can directly install LoftNN into your virtual environment using:
```
pip install .
```

## Model training scripts

LoftNN is a framework used as part of model training scripts. Two model training scripts are provided:

- [cnn-ec](https://github.com/m-maresch/cnn-ec) for CNNs (ResNets) and ViTs (Swin Transformers)
- [nanoGPT-ec](https://github.com/m-maresch/nanoGPT-ec) for GPT models

## Local training

To get started with training a model locally using LoftNN, we recommend using [cnn-ec](https://github.com/m-maresch/cnn-ec) with the small ResNet18 model.

The following has been tested on a MacBook Pro (Intel-based) with 16 GB RAM and using Python 3.10.18.

### cnn-ec

To train the small ResNet18 model, clone the [cnn-ec](https://github.com/m-maresch/cnn-ec) repository, then run the following with the venv:
```
cd cnn-ec

# Prepare the data
curl -O https://download.pytorch.org/tutorial/hymenoptera_data.zip
unzip hymenoptera_data.zip -d data

pip install torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu

# Start training
torchrun --nproc_per_node=3 train.py --device=cpu --parallelism="hybrid" --num_epochs=5 --planner="ecs" --no_ambp --model="resnet18"
```

This trains ResNet18 with 3 processes using LoftNNs hybrid pipeline parallelism. The configuration is determined by ECSP. See the paper for details.

Note: If timeout errors occur during training, try increasing the timeout by running e.g. `export TIMEOUT_SECONDS=120` before the `torchrun` command.

### nanoGPT-ec

To do the same with a small GPT model, clone the [nanoGPT-ec](https://github.com/m-maresch/nanoGPT-ec) repository, then run the following with the venv:
```
cd nanoGPT-ec

# Prepare the data
curl -LJO https://huggingface.co/datasets/Skylion007/openwebtext/resolve/39d1a6d3f3a4c838b27bbc65bfb14a2f6f727740/subsets/urlsf_subset01.tar?download=true
mv urlsf_subset01.tar data/openwebtext/
python data/openwebtext/prepare.py # takes a few minutes

# Start training
torchrun --nproc_per_node=3 train.py config/train_openwebtext_char_small.py --device=cpu --max_iters=5 --parallelism="hybrid" --planner="ecs" --use_ambp=False
```

## Distributed training

The following environment variables need to be set on the nodes involved in the distributed training for LoftNN to be available:
- `RANK`, e.g. 0, 1, 2 (varies per node)
- `LOCAL_RANK`, e.g. 0, 1, 2 (varies per node)
- `WORLD_SIZE`, e.g. 3
- `MASTER_ADDR`, e.g. 100.90.80.70
- `MASTER_PORT`, e.g. 5000

## Usage

LoftNN is used as part of model training scripts as follows:
```
import loftnn

from loftnn import (
    DataParallel,
    PipelineParallel,
    HybridPipelineParallel
)
from loftnn.types import Device

if loftnn.is_available():
    process_config = loftnn.ProcessConfiguration.from_env()

if use_data_parallelism:
    dist_model = DataParallel(
        model=model,
        process_config=process_config,
        device=Device.cuda,
    )
elif use_pipeline_parallelism:
    microbatch_sample = X.chunk(num_microbatches)[0]
    pipeline_config = loftnn.PipelineConfiguration(
        split_points=split_points,
        num_microbatches=num_microbatches,
        microbatch_sample=microbatch_sample,
        loss_fn=loss_fn,
    )

    dist_model = PipelineParallel(
        model,
        process_config=process_config,
        pipeline_config=pipeline_config,
        device=Device.cuda,
    )
elif use_hybrid_pipeline_parallelism:
    microbatch_sample = X.chunk(num_microbatches)[0]
    pipeline_config = loftnn.HybridPipelineConfiguration(
        planner=planner, # 'exact' or 'ecs'
        num_microbatches=num_microbatches,
        microbatch_sample=microbatch_sample,
        loss_fn=loss_fn,
    )

    dist_model = HybridPipelineParallel(
        model,
        process_config=process_config,
        hybrid_pipeline_config=pipeline_config,
        device=Device.cuda,
    )

    (
        split_points,
        device_groups,
        samples_allocated,
        activation_checkpointing_budgets,
    ) = dist_model.compute_plan()

    dist_model.prepare_schedule(
        split_points, device_groups, samples_allocated,
        activation_checkpointing_budgets
    )
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details. Third-party library notices are documented in `THIRD-PARTY-NOTICES.txt`.

## Dependencies

Thanks to everyone contributing to any of the following projects:

- PyTorch
- torchinfo
- torchview
- NumPy
- Matplotlib
- psutil
- NetworkX
- nanoGPT

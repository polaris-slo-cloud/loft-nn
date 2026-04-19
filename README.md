# LoftNN: Distributed Training of Deep Learning Models in the Edge-Cloud-Space Continuum

LoftNN is a framework for distributed training in the edge-cloud-space continuum.

All details on LoftNN can be found in the [paper](TODO).

Publication: TODO ([TODO](TODO))

Abstract:
>
>
>

## Model training scripts

Two model training scripts are provided:

- [nanoGPT-ec](https://github.com/m-maresch/nanoGPT-ec) for GPT models
- [cnn-ec](https://github.com/m-maresch/cnn-ec) for CNNs (ResNet50) and ViTs (Swin Transformers)

## Installation and development setup

Clone the repository, then install LoftNN into your virtual environment:
```
pip install -e .
```

## Example usage

The following environment variables need to be set for LoftNN to be available for distributed training:
- `RANK`, e.g. 0, 1, 2 (varies per node)
- `LOCAL_RANK`, e.g. 0, 1, 2 (varies per node)
- `WORLD_SIZE`, e.g. 3
- `MASTER_ADDR`, e.g. 100.90.80.70
- `MASTER_PORT`, e.g. 5000

Then, LoftNN can be used as part of model training scripts as follows:
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

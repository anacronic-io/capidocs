Configuration Documentation
===========================

This document provides an overview of the configuration settings used in the CapibaraGPT model. The configurations are defined in `config.py` and are essential for setting up the model's environment and parameters.

Environment Variables
---------------------

The following environment variables are loaded from a `.env` file using `python-dotenv`:

- **RANDOM_SEED**: Seed for random number generation.
- **CAPIBARA_LOG_LEVEL**: Logging level for the application.
- **CAPIBARA_BASE_CONFIG_PATH**: Path to the base configuration file.
- **GCS_BUCKET_NAME**: Name of the Google Cloud Storage bucket.

These variables are accessed in the code as follows:

```python
import os
RANDOM_SEED = os.getenv('RANDOM_SEED')
CAPIBARA_LOG_LEVEL = os.getenv('CAPIBARA_LOG_LEVEL')
CAPIBARA_BASE_CONFIG_PATH = os.getenv('CAPIBARA_BASE_CONFIG_PATH')
GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME')
```

Model Configuration Classes
---------------------------

### OptimizationConfig

Defines the optimization settings for the model.

- **use_flash_attention**: Enable flash attention (default: `True`).
- **use_rotary_embeddings**: Use rotary embeddings (default: `True`).
- **use_alibi**: Use ALiBi (Attention with Linear Biases) (default: `True`).
- **use_parallel_layers**: Use parallel layers (default: `True`).
- **attention_implementation**: Type of attention implementation (default: `"flash"`).
- **use_mixed_precision**: Enable mixed precision training (default: `True`).
- **mixed_precision_dtype**: Data type for mixed precision (default: `"bfloat16"`).

### TPUConfig

Configuration specific to TPU usage.

- **use_tpu**: Enable TPU usage (default: `True`).
- **tpu_name**: Name of the TPU (default: `"capibara-tpu"`).
- **tpu_zone**: Zone of the TPU (default: `"us-central1-a"`).
- **gcp_project**: Google Cloud Project name (default: `"capibara-project"`).
- **tpu_topology**: Topology of the TPU (default: `"v3-8"`).
- **num_tpu_cores**: Number of TPU cores (default: `8`).

### TrainingConfig

Defines the training parameters for the model.

- **batch_size**: Size of the training batch (default: `32`).
- **learning_rate**: Learning rate for the optimizer (default: `5e-5`).
- **num_train_steps**: Number of training steps (default: `100000`).
- **num_warmup_steps**: Number of warmup steps (default: `1000`).
- **optimizer**: Optimizer type (default: `"adamw"`).
- **weight_decay**: Weight decay for the optimizer (default: `0.01`).
- **gradient_accumulation_steps**: Steps for gradient accumulation (default: `1`).
- **max_grad_norm**: Maximum gradient norm (default: `1.0`).
- **seed**: Random seed for training (default: `42`).

Usage
-----

To use these configurations, ensure that the `.env` file is correctly set up in your project directory. The configurations can be accessed and modified in your Python code as needed.

```python
from capibara_model.core.config import OptimizationConfig, TPUConfig, TrainingConfig

# Example usage
opt_config = OptimizationConfig()
tpu_config = TPUConfig()
train_config = TrainingConfig()

print(f"Batch size: {train_config.batch_size}")
```

This document should be updated regularly to reflect any changes in the configuration settings or the addition of new parameters.
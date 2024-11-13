Configuration Module
==================

Overview
--------
The configuration module provides a unified interface for managing CapibaraModel settings through environment variables, configuration files, and Weights & Biases integration.

Key Features
-----------
- Environment variable loading with type conversion
- .env file support
- Weights & Biases integration
- Default value handling
- Configuration validation
- Dataclass-based configuration structure

Configuration Classes
-------------------

CapibaraConfig
~~~~~~~~~~~~~
.. autoclass:: capibara_model.core.config.CapibaraConfig
   :members:
   :undoc-members:
   :special-members: __init__

Main configuration container that holds all sub-configurations:

- Training configuration
- Model parameters
- Pruning settings
- Weights & Biases integration

TrainingConfig
~~~~~~~~~~~~
.. autoclass:: capibara_model.core.config.TrainingConfig
   :members:
   :undoc-members:

Training-specific parameters including:

- Random seed
- Batch size
- Learning rate
- Number of epochs
- Optimizer selection
- Loss function
- Metrics
- Early stopping patience

ModelConfig
~~~~~~~~~~
.. autoclass:: capibara_model.core.config.ModelConfig
   :members:
   :undoc-members:

Model architecture parameters:

- Input dimension
- Hidden size
- Number of layers
- Dropout rate
- Activation function

PruningConfig
~~~~~~~~~~~
.. autoclass:: capibara_model.core.config.PruningConfig
   :members:
   :undoc-members:

Model pruning parameters:

- MOR threshold
- Sparsity ratio
- Pruning method
- Pruning schedule

WandbConfig
~~~~~~~~~~
.. autoclass:: capibara_model.core.config.WandbConfig
   :members:
   :undoc-members:

Weights & Biases integration settings:

- Project name
- Entity
- Model logging
- Gradient logging

Usage Examples
------------

Loading from Environment Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from capibara_model.core.config import CapibaraConfig

    # Load configuration from environment variables
    config = CapibaraConfig.from_env()

Loading from Dictionary
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    config_dict = {
        "training": {
            "batch_size": 64,
            "learning_rate": 0.001
        },
        "model": {
            "hidden_size": 1024
        }
    }
    config = CapibaraConfig.from_dict(config_dict)

Environment Variables
------------------

Training Configuration
~~~~~~~~~~~~~~~~~~~~
- TRAINING_SEED (default: 42)
- TRAINING_BATCH_SIZE (default: 32)
- TRAINING_LEARNING_RATE (default: 0.001)
- TRAINING_NUM_EPOCHS (default: 10)
- TRAINING_OPTIMIZER (default: 'adam')
- TRAINING_LOSS_FUNCTION (default: 'cross_entropy')
- TRAINING_METRICS
- TRAINING_EARLY_STOPPING_PATIENCE (default: 5)

Model Configuration
~~~~~~~~~~~~~~~~
- MODEL_INPUT_DIM (default: 768)
- MODEL_HIDDEN_SIZE (default: 768)
- MODEL_NUM_LAYERS (default: 12)
- MODEL_DROPOUT_RATE (default: 0.1)
- MODEL_ACTIVATION_FUNCTION (default: 'relu')

Pruning Configuration
~~~~~~~~~~~~~~~~~~
- PRUNING_MOR_THRESHOLD (default: 0.7)
- PRUNING_SPARSITY_RATIO (default: 0.5)
- PRUNING_METHOD (default: 'magnitude')
- PRUNING_SCHEDULE (default: 'constant')

Weights & Biases Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~
- WANDB_PROJECT
- WANDB_ENTITY
- WANDB_LOG_MODEL (default: false)
- WANDB_LOG_GRADIENTS (default: false)

Validation
---------
The configuration system includes validation for:

- Dropout rate (must be between 0 and 1)
- Sparsity ratio (must be between 0 and 1)
- MOR threshold (must be between 0 and 1)
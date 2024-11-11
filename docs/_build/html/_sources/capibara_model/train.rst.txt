Training Module
=============

Overview
--------
The training module provides functionality for training CapibaraGPT models using JAX and Flax. It includes support for distributed training on TPUs, model checkpointing, early stopping, and model export to ONNX format.

Core Features
------------
- Distributed training on TPU/GPU
- Mixed precision training
- Gradient accumulation
- Early stopping
- Model checkpointing
- ONNX export
- Google Cloud Storage integration
- Efficient data loading
- Training state management

Architecture
-----------

Training State
~~~~~~~~~~~~

.. code-block:: python

    def create_train_state(rng, model, learning_rate, config):
        """Initializes training state with optimizer and model."""
        params = model.init(rng, jnp.ones((1, config.input_dim)),
                          model.init_hidden_state(1))['params']
        tx = optax.adamw(learning_rate)
        return train_state.TrainState.create(
            apply_fn=model.apply, 
            params=params, 
            tx=tx
        )

Training Steps
~~~~~~~~~~~~

.. code-block:: python

    def train_step(state, batch, hidden_state, dropout_rng):
        """Single training step with forward and backward pass."""
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
        
        def loss_fn(params):
            return state.apply_fn(
                {'params': params}, 
                batch['input_ids'], 
                hidden_state, 
                rngs={'dropout': dropout_rng}
            ).mean()

Configuration
-----------

Training Settings
~~~~~~~~~~~~~~

.. code-block:: yaml

    training:
      batch_size: 32
      learning_rate: 2e-5
      num_epochs: 100
      early_stopping_patience: 5
      checkpoint_frequency: 10
      seed: 42
      mixed_precision: true
      gradient_accumulation_steps: 8

Optimization
~~~~~~~~~~

.. code-block:: yaml

    optimizer:
      name: "adamw"
      beta1: 0.9
      beta2: 0.999
      epsilon: 1e-8
      weight_decay: 0.01

Usage Examples
------------

Basic Training
~~~~~~~~~~~~

.. code-block:: python

    from capibara_model.train import main as train_main
    
    # Train with default config
    train_main()

Custom Configuration
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from capibara_model.core.config import CapibaraConfig
    from capibara_model.train import main as train_main
    
    # Load custom config
    config = CapibaraConfig.from_yaml("custom_config.yaml")
    
    # Train with custom config
    train_main(config)

Implementation Details
-------------------

Data Loading
~~~~~~~~~~
- Efficient data loading with datasets library
- Data sharding for distributed training
- Dynamic batching
- Memory-efficient data streaming

Model Export
~~~~~~~~~~
- ONNX format export
- Google Cloud Storage upload
- Checkpoint management
- State serialization

Performance Optimizations
----------------------

Memory Management
~~~~~~~~~~~~~~
- Gradient accumulation
- Mixed precision training
- Weight pruning
- Efficient state updates

Distributed Training
~~~~~~~~~~~~~~~~~
- TPU/GPU support
- Data parallelism
- Gradient synchronization
- Device placement optimization

See Also
--------
- :doc:`core/model`: Model architecture documentation
- :doc:`core/config`: Configuration system
- :doc:`utils/checkpoints`: Checkpoint utilities

Notes
-----
- Monitor memory usage during training
- Adjust batch size based on available hardware
- Consider gradient accumulation for large models
- Test checkpoint restoration 
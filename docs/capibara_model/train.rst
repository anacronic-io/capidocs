Training Module
==============

Overview
--------
The training module implements the core training loop and model conversion pipeline for CapibaraModel using JAX and TPU optimization.

Key Components
-------------

Training Pipeline
~~~~~~~~~~~~~~~
.. automodule:: capibara_model.train
   :members:
   :undoc-members:

Features:
- JAX-based training optimization
- TPU-specific optimizations
- Automatic mixed precision (bfloat16)
- Model conversion to TensorFlow/TFLite
- Integrated logging and monitoring

Core Functions
-------------

Training Loop
~~~~~~~~~~~~
.. autofunction:: capibara_model.train.train_step

Handles single training step optimization:
- Gradient computation and updates
- Loss calculation
- Parameter updates
- RNG key management

State Management
~~~~~~~~~~~~~~
.. autofunction:: capibara_model.train.create_train_state

Manages training state including:
- Parameter initialization
- Optimizer configuration (AdamW)
- Learning rate scheduling
- Batch size configuration

Model Export
~~~~~~~~~~~
Supports multiple export formats:
- JAX serialized parameters
- TensorFlow SavedModel
- TFLite optimized model
- Quantization support

TPU Optimization
--------------

Hardware Configuration
~~~~~~~~~~~~~~~~~~~~
- TPU-specific memory management
- Multi-core processing support
- Batch processing optimization
- Mixed precision training

Performance Features
~~~~~~~~~~~~~~~~~~
- bfloat16 automatic mixed precision
- Gradient checkpointing
- Memory-efficient training
- TPU Pod support

Usage Example
-----------

.. code-block:: python

    from capibara_model.train import main
    from capibara_model.core.config import CapibaraConfig

    # Load configuration
    config = CapibaraConfig.from_yaml('config.yaml')

    # Start training
    main()

Configuration
------------

Training Parameters
~~~~~~~~~~~~~~~~~
- Batch size
- Learning rate
- Number of epochs
- Optimizer settings
- Loss function
- Early stopping

Model Parameters
~~~~~~~~~~~~~~
- Input dimension
- Hidden size
- Number of layers
- Dropout rate
- Activation functions

Export Settings
~~~~~~~~~~~~~
- SavedModel configuration
- TFLite optimization
- Quantization options
- Pruning settings

TPU Configuration
~~~~~~~~~~~~~~~
- Device placement
- Memory management
- Precision settings
- Multi-core utilization

See Also
--------
- :doc:`config`
- :doc:`model`
- :doc:`inference`
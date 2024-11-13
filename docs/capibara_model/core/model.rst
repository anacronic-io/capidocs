Model Core
=========

The core model implementation for CapibaraGPT, implementing an advanced language model using JAX and Flax.

Overview
--------
.. automodule:: capibara_model.core.model
   :members:
   :undoc-members:
   :show-inheritance:

Classes
-------

CapibaraModel
~~~~~~~~~~~~
.. autoclass:: capibara_model.core.model.CapibaraModel
   :members:
   :undoc-members:
   :special-members: __init__, __call__

   Main model class that integrates multiple NLP components including byte processing,
   synthetic embeddings, and various neural processing layers.

Component Classes
~~~~~~~~~~~~~~~

BitnetLiquidBlock
----------------
.. autoclass:: capibara_model.core.model.BitnetLiquidBlock
   :members:
   :undoc-members:
   :special-members: __init__, __call__

EmbeddingLayer
-------------
.. autoclass:: capibara_model.core.model.EmbeddingLayer
   :members:
   :undoc-members:
   :special-members: __init__, __call__

AttentionLayer
-------------
.. autoclass:: capibara_model.core.model.AttentionLayer
   :members:
   :undoc-members:
   :special-members: __init__, __call__

BitnetLayer
----------
.. autoclass:: capibara_model.core.model.BitnetLayer
   :members:
   :undoc-members:
   :special-members: __init__, __call__

OutputLayer
----------
.. autoclass:: capibara_model.core.model.OutputLayer
   :members:
   :undoc-members:
   :special-members: __init__, __call__

Functions
---------

Model Creation and Management
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: capibara_model.core.model.create_large_capibara_model
.. autofunction:: capibara_model.core.model.create_train_state
.. autofunction:: capibara_model.core.model.count_parameters

Training & Evaluation
~~~~~~~~~~~~~~~~~~
.. autofunction:: capibara_model.core.model.train_step
.. autofunction:: capibara_model.core.model.eval_step
.. autofunction:: capibara_model.core.model.main

Dependencies
-----------

Core Libraries:
~~~~~~~~~~~~~
- JAX and JAX.numpy
- Flax and Flax.linen
- Optax
- NumPy
- ONNX and ONNX Runtime

Custom Components:
~~~~~~~~~~~~~~~
- CapibaraConfig
- BitNet
- Platonic
- GameTheory
- SelfAttention
- SparseCapibara
- SyntheticEmbedding
- BitNetQuantizer
- MixtureOfRookies

Submodels:
~~~~~~~~~
- AlephTilde
- CapibaraByte
- CapibaraJAXSSM
- Capibara2
- Liquid
- MetaBAMDP
- SNNSLiCell
- SpikeSSM

Example Usage
------------
.. code-block:: python

    from capibara_model.core.model import CapibaraModel, create_large_capibara_model
    from capibara_model.core.config import CapibaraConfig

    # Load configuration
    config = CapibaraConfig.from_yaml('config.yaml')

    # Create model
    model, params = create_large_capibara_model(config)

    # Initialize training state
    state = create_train_state(rng, model, learning_rate=1e-4, config=config)

    # Training loop
    for batch in data_loader:
        state, loss, hidden_state, rng = train_step(state, batch, hidden_state, rng)
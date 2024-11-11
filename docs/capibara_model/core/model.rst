Model Core
=========

The core model implementation for CapibaraGPT.

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

BitnetLiquidBlock
~~~~~~~~~~~~~~~
.. autoclass:: capibara_model.core.model.BitnetLiquidBlock
   :members:
   :undoc-members:
   :special-members: __init__, __call__

Functions
---------

Model Creation
~~~~~~~~~~~~
.. autofunction:: capibara_model.core.model.create_large_capibara_model
.. autofunction:: capibara_model.core.model.create_train_state

Training & Evaluation
~~~~~~~~~~~~~~~~~~
.. autofunction:: capibara_model.core.model.train_step
.. autofunction:: capibara_model.core.model.eval_step
.. autofunction:: capibara_model.core.model.apply_pruning

Utilities
~~~~~~~~
.. autofunction:: capibara_model.core.model.count_parameters
.. autofunction:: capibara_model.core.model.main

Dependencies
-----------
- JAX
- Flax
- ONNX
- NumPy
- Optax

Example Usage
------------
.. code-block:: python

    from capibara_model.core.model import CapibaraModel, create_large_capibara_model
    from capibara_model.core.config import CapibaraConfig

    # Load configuration
    config = CapibaraConfig.from_yaml('config.yaml')

    # Create model
    model, params = create_large_capibara_model(config) 
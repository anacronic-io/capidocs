Meta BAMDP Module
================

.. module:: meta_bamdp
   :synopsis: Implementation of a Meta BAMDP layer for neural networks using JAX/Flax.

This module implements a Meta BAMDP layer that applies meta-learning operations to input data efficiently.

Classes
-------

BaseLayer
~~~~~~~~~

.. autoclass:: BaseLayer
   :members:
   :undoc-members:
   :show-inheritance:

   Base class for all Capibara layers.

MetaBAMDP
~~~~~~~~~

.. autoclass:: MetaBAMDP
   :members:
   :undoc-members:
   :show-inheritance:

   A neural network layer implementing the Meta BAMDP architecture.

   .. rubric:: Attributes

   - **input_dim** (``int``): Input dimension
   - **hidden_dim** (``int``): Hidden state dimension  
   - **output_dim** (``int``): Output dimension
   - **dropout_rate** (``float``): Dropout rate for regularization (default: 0.1)

   .. rubric:: Methods

   .. automethod:: setup

      Initializes the layer components.

   .. automethod:: __call__

      Forward pass of the layer.

Implementation Details
--------------------

The Meta BAMDP layer implements:

1. Dense transformation to hidden dimension
2. Dropout for regularization
3. Dense projection to output dimension
4. Layer normalization

The implementation uses JAX's automatic differentiation and Flax's neural network modules.

Dependencies
-----------

- JAX and JAX NumPy for array operations
- Flax for neural network layers
- Logging for operation tracking

Example Usage
------------

.. code-block:: python

    # Initialize the layer
    layer = MetaBAMDP(
        input_dim=256,
        hidden_dim=512, 
        output_dim=256,
        dropout_rate=0.1
    )

    # Create sample input
    batch_size, seq_len, input_dim = 32, 128, 256
    x = jax.random.normal(key, (batch_size, seq_len, input_dim))

    # Initialize parameters
    params = layer.init(key, x, training=True)

    # Forward pass
    output = layer.apply(params, x, training=True)

See Also
--------

- :doc:`capibara_jax_ssm`
- :doc:`capibara_byte`
- :doc:`liquid` 
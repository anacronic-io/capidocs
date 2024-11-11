Liquid Module
============

.. module:: liquid
   :synopsis: Implementation of a Liquid layer for neural networks using JAX/Flax.

This module implements a Liquid layer that combines linear transformations, GELU activation, 
and layer normalization for efficient processing of input arrays.

Classes
-------

BaseLayer
~~~~~~~~~

.. autoclass:: BaseLayer
   :members:
   :undoc-members:
   :show-inheritance:

   Base class for all Capibara layers.

Liquid
~~~~~~

.. autoclass:: Liquid
   :members:
   :undoc-members:
   :show-inheritance:

   A neural network layer implementing the Liquid architecture.

   .. rubric:: Attributes

   - **dim** (``int``): Input and output dimension
   - **expansion_factor** (``int``): Factor for dimension expansion (default: 4)
   - **dropout_rate** (``float``): Dropout rate for regularization (default: 0.1)
   - **use_expansion** (``bool``): Whether to use dimension expansion (default: True)

   .. rubric:: Methods

   .. automethod:: __post_init__

      Validates initialization parameters.

   .. automethod:: __call__

      Forward pass of the layer.

      Args:
          x (jnp.ndarray): Input array of shape (batch_size, sequence_length, dim)
          training (bool): Whether in training mode
      
      Returns:
          jnp.ndarray: Output array of shape (batch_size, sequence_length, dim)

   .. automethod:: get_config

      Returns layer configuration as a dictionary.

Implementation Details
--------------------

The layer implements the following sequence of operations:

1. Input validation
2. Optional dimension expansion via dense layer
3. GELU activation
4. Dropout
5. Dense projection back to original dimension
6. Layer normalization
7. Residual connection

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
    layer = Liquid(
        dim=256,
        expansion_factor=4,
        dropout_rate=0.1,
        use_expansion=True
    )

    # Create sample input
    batch_size, seq_len, dim = 32, 128, 256
    x = jax.random.normal(key, (batch_size, seq_len, dim))

    # Initialize parameters
    params = layer.init(key, x, training=True)

    # Forward pass
    output = layer.apply(params, x, training=True)

See Also
--------

- :doc:`capibara_byte`
- :doc:`capibara_jax_ssm`
- :doc:`meta_bamdp` 
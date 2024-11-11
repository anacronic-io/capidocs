Capibara Byte Module
===================

This module implements a Mamba Byte layer for neural networks using JAX/Flax, combining advanced techniques like BitNet, Liquid layers, and CapibaraJAXSSM.

.. module:: capibara_byte

Classes
-------

CapibaraByte
~~~~~~~~~~~

.. autoclass:: CapibaraByte
   :members:
   :undoc-members:
   :show-inheritance:

   A neural network layer that combines various techniques for efficient processing of byte-level inputs.

   .. rubric:: Attributes

   - **dim** (``int``): Input and output dimension
   - **num_bitnet_layers** (``int``): Number of BitNet layers
   - **num_liquid_layers** (``int``): Number of Liquid layers
   - **dropout_rate** (``float``): Dropout rate for regularization (default: 0.1)
   - **use_residual** (``bool``): Whether to use residual connections (default: True)

   .. rubric:: Methods

   .. automethod:: setup

      Initializes the layer components including:
      
      - CapibaraJAXSSM layers
      - Synthetic embedding
      - BitNet layers
      - Liquid layers
      - BitNet quantizer
      - Meta BAMDP layer
      - Final dense layer

   .. automethod:: __call__

      Forward pass of the layer.

      Args:
          x (jnp.ndarray): Input array of shape (batch_size, seq_len, dim)
          training (bool): Whether in training mode
      
      Returns:
          jnp.ndarray: Output array of shape (batch_size, seq_len, dim)

   .. automethod:: _validate_input

      Validates input array dimensions.

   .. automethod:: get_config

      Returns layer configuration as a dictionary.

Implementation Details
--------------------

The layer implements a sequence of operations:

1. Input validation
2. CapibaraJAXSSM processing
3. Synthetic embedding
4. BitNet and Liquid layer sequence
5. Second CapibaraJAXSSM pass
6. BitNet quantization
7. Meta BAMDP processing
8. Final dense projection
9. Optional residual connection
10. Dropout

Dependencies
-----------

- JAX and Flax for neural network operations
- Custom layers:
    - SyntheticEmbedding
    - BitNet
    - Liquid
    - BitNetQuantizer
    - MetaBAMDP
    - CapibaraJAXSSM

Example Usage
------------

.. code-block:: python

    # Initialize the layer
    layer = CapibaraByte(
        dim=256,
        num_bitnet_layers=3,
        num_liquid_layers=3,
        dropout_rate=0.1,
        use_residual=True
    )

    # Create sample input
    batch_size, seq_len, dim = 32, 10, 256
    x = jax.random.normal(key, (batch_size, seq_len, dim))

    # Initialize parameters
    params = layer.init(key, x, training=True)

    # Forward pass
    output = layer.apply(params, x, training=True)

See Also
--------

- :doc:`capibara_jax_ssm`
- :doc:`bitnet`
- :doc:`liquid`
- :doc:`meta_bamdp` 
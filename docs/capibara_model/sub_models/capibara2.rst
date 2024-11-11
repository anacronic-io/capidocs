Capibara2 Module
===============

.. module:: capibara2
   :synopsis: Implementation of a recurrent layer using JAX/Flax with SSM capabilities.

This module implements a recurrent layer for neural networks using JAX/Flax, specifically designed to work with State Space Models (SSM).

Classes
-------

BaseLayer
~~~~~~~~~

.. autoclass:: BaseLayer
   :members:
   :undoc-members:
   :show-inheritance:

   Base class for all Capibara layers.

Capibara2
~~~~~~~~~

.. autoclass:: Capibara2
   :members:
   :undoc-members:
   :show-inheritance:

   A neural network layer that implements a recurrent operation using CapibaraJAXSSM.

   .. rubric:: Attributes

   - **input_dim** (``int``): Input dimension
   - **hidden_dim** (``int``): Hidden state dimension
   - **output_dim** (``int``): Output dimension
   - **conv_dim** (``int``): Convolutional dimension (default: 4)

   .. rubric:: Methods

   .. automethod:: setup

      Initializes the CapibaraJAXSSM layer with configured dimensions.

   .. automethod:: __call__

      Forward pass of the layer.

      Args:
          x (jnp.ndarray): Input array of shape (batch_size, seq_len, input_dim)
      
      Returns:
          jnp.ndarray: Output array of shape (batch_size, seq_len, output_dim)

   .. automethod:: _validate_input

      Validates input array dimensions and logs any errors.

   .. automethod:: get_config

      Returns layer configuration as a dictionary.

Implementation Details
--------------------

The layer implements the following sequence of operations:

1. Input validation
2. Zero state initialization
3. JAX scan over sequence length
4. CapibaraJAXSSM forward pass
5. Output transposition

Dependencies
-----------

- JAX and JAX NumPy for array operations
- Flax for neural network layers
- Custom layers:
    - CapibaraJAXSSM
    - BaseLayer

Example Usage
------------

.. code-block:: python

    # Initialize the layer
    layer = Capibara2(
        input_dim=256,
        hidden_dim=512,
        output_dim=256,
        conv_dim=4
    )

    # Create sample input
    batch_size, seq_len, input_dim = 32, 10, 256
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, seq_len, input_dim))

    # Initialize parameters
    params = layer.init(key, x)

    # Forward pass
    output = layer.apply(params, x)

See Also
--------

- :doc:`capibara_jax_ssm`
- :doc:`capibara_byte`
- :doc:`liquid` 
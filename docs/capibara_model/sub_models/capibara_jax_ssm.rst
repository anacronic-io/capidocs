Capibara JAX SSM Module
======================

This module implements a State Space Model (SSM) using JAX and Flax for efficient computation and training.

.. module:: capibara_jax_ssm

Classes
-------

CapibaraJAXSSM
~~~~~~~~~~~~~

.. autoclass:: CapibaraJAXSSM
   :members:
   :undoc-members:
   :show-inheritance:

   A neural network layer implementing a State Space Model with JAX/Flax.

   .. rubric:: Attributes

   - **config** (:class:`CapibaraConfig`): Model configuration
   - **dropout_rate** (``float``): Dropout rate for regularization (default: 0.1)

   .. rubric:: Methods

   .. automethod:: __call__

      Forward pass of the layer.

      Args:
          x (jnp.ndarray): Input array of shape (batch_size, seq_len, input_dim)
          state (jnp.ndarray): Initial state
          deterministic (bool): Whether in training mode
      
      Returns:
          jnp.ndarray: Output array of shape (batch_size, seq_len, output_dim)

   .. automethod:: causal_conv

      Performs causal convolution with SiLU activation.

      Args:
          x (jnp.ndarray): Input array
          conv_kernel (jnp.ndarray): Convolution kernel
      
      Returns:
          jnp.ndarray: Convolved and activated output

   .. automethod:: ssm_step

      Single step of the State Space Model.

      Args:
          carry: Tuple with hidden state and previous output
          x_t: Input at time t
          A, B, C: SSM parameters
      
      Returns:
          Tuple of new state and output

Helper Functions
--------------

hippo_initialization
~~~~~~~~~~~~~~~~~~

.. autofunction:: hippo_initialization

   Initializes the A matrix using HiPPO method.

   Args:
       N (int): State dimension
   
   Returns:
       jnp.ndarray: Initialized A matrix

Implementation Details
--------------------

The SSM implementation includes:

1. HiPPO initialization for matrix A
2. Causal convolution with SiLU activation
3. State space model computation
4. Dropout for regularization
5. Residual connections
6. Parameter initialization using normal distribution

Training Configuration
--------------------

The model supports training with:

- Weight decay regularization
- Adam optimizer
- Multi-device training using JAX pmap
- Batch processing
- Loss function with L2 regularization

Dependencies
-----------

- JAX and Flax for neural network operations
- NumPy for numerical operations
- Optax for optimization
- Custom configurations via CapibaraConfig

Example Usage
------------

.. code-block:: python

    # Initialize the model
    config = {
        'state_dim': 32,
        'input_dim': 10,
        'output_dim': 5,
        'conv_dim': 4,
    }
    model = CapibaraJAXSSM(config=config, dropout_rate=0.1)

    # Create sample input
    batch_size = 32
    x_data = jax.random.normal(key, (batch_size, config['input_dim']))
    initial_state = jnp.zeros((batch_size, config['state_dim']))

    # Forward pass
    output = model.apply({'params': params}, x_data, initial_state)

See Also
--------

- :doc:`capibara_byte`
- :doc:`liquid`
- :doc:`meta_bamdp` 
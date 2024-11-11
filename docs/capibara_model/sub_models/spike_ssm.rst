Spike SSM Module
===============

.. module:: spike_ssm
   :synopsis: Implementation of a Spike-SSM layer for neural networks using JAX/Flax.

This module implements a Spike-SSM (State Space Model) layer that combines spiking neural dynamics with state space models for efficient sequence processing.

Classes
-------

SpikeSSMState
~~~~~~~~~~~~

.. autoclass:: SpikeSSMState
   :members:
   :undoc-members:
   :show-inheritance:

   Named tuple that holds the state of a Spike-SSM layer.

   .. rubric:: Attributes

   - **hidden_state** (jnp.ndarray): Hidden state of the SSM
   - **output_state** (jnp.ndarray): Output state after spiking dynamics

SpikeSSM
~~~~~~~~

.. autoclass:: SpikeSSM
   :members:
   :undoc-members:
   :show-inheritance:

   Implementation of a Spike-SSM layer combining state space models with spiking dynamics.

   .. rubric:: Attributes

   - **input_dim** (int): Input dimension
   - **output_dim** (int): Output dimension
   - **hidden_dim** (int): Hidden state dimension
   - **sparsity_level** (float): Sparsity level for connections (default: 0.8)
   - **tau** (float): Time constant for neural dynamics (default: 10.0)
   - **dt** (float): Time step for simulation (default: 1.0)
   - **noise_std** (float): Standard deviation of noise (default: 0.01)

Implementation Details
--------------------

The Spike-SSM layer implements:

1. Linear transformations for input, hidden and output
2. Spiking neural dynamics with customizable time constants
3. Sparse connectivity through random masks
4. Noise injection for regularization
5. State management for sequential processing

The implementation uses JAX for automatic differentiation and Flax for neural network modules.

Dependencies
-----------

- JAX and JAX NumPy for array operations
- Flax for neural network layers
- Logging for operation tracking

Example Usage
------------

.. code-block:: python

    # Initialize the layer
    layer = SpikeSSM(
        input_dim=256,
        output_dim=128,
        hidden_dim=512,
        sparsity_level=0.8,
        tau=10.0,
        dt=1.0,
        noise_std=0.01
    )

    # Create sample input and state
    batch_size, seq_len, input_dim = 32, 100, 256
    x = jax.random.normal(key, (batch_size, seq_len, input_dim))
    state = SpikeSSMState(
        hidden_state=jnp.zeros((batch_size, hidden_dim)),
        output_state=jnp.zeros((batch_size, output_dim))
    )

    # Initialize parameters
    params = layer.init(key, x, state)

    # Forward pass
    output, new_state = layer.apply(params, x, state)

See Also
--------

- :doc:`capibara_jax_ssm`
- :doc:`snns_LiCell`
- :doc:`liquid` 
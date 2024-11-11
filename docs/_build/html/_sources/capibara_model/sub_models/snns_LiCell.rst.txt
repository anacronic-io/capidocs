Spiking Neural Network State (SNNS) Module
========================================

.. module:: snns_LiCell
   :synopsis: Implementation of Leaky Integrate-and-Fire neurons using JAX/Flax.

This module implements Spiking Neural Networks using Leaky Integrate-and-Fire (LIF) neurons with JAX and Flax.

Classes
-------

LIFState
~~~~~~~~

.. autoclass:: LIFState
   :members:
   :undoc-members:
   :show-inheritance:

   Named tuple that holds the state of a LIF neuron.

   .. rubric:: Attributes

   - **v** (jnp.ndarray): Membrane potential
   - **spikes** (jnp.ndarray): Spike history
   - **adaptive_thresh** (jnp.ndarray): Adaptive threshold

LIFCell
~~~~~~~

.. autoclass:: LIFCell
   :members:
   :undoc-members:
   :show-inheritance:

   Implementation of a Leaky Integrate-and-Fire neuron.

   .. rubric:: Attributes

   - **tau** (float): Time constant (default: 20.0)
   - **v_rest** (float): Resting potential (default: -65.0)
   - **v_reset** (float): Reset potential (default: -70.0)
   - **threshold** (float): Firing threshold (default: -50.0)
   - **adaptive_threshold** (bool): Use adaptive threshold (default: False)
   - **threshold_adaptation_tau** (float): Adaptation time constant (default: 100.0)

SNNS
~~~~

.. autoclass:: SNNS
   :members:
   :undoc-members:
   :show-inheritance:

   Spiking Neural Network layer using LIF neurons.

   .. rubric:: Attributes

   - **input_dim** (int): Input dimension
   - **hidden_dim** (int): Number of LIF neurons
   - **lif_params** (dict): Parameters for LIF neurons
   - **activation** (Callable): Activation function

Implementation Details
--------------------

The implementation includes:

1. LIF neuron dynamics with adaptive threshold
2. Dense layer for input transformation
3. Scan over sequence length
4. Spike generation and reset
5. State management

Example Usage
------------

.. code-block:: python

    # Initialize the layer
    lif_params = {
        "tau": 20.0,
        "v_rest": -65.0,
        "v_reset": -70.0,
        "threshold": -50.0,
        "adaptive_threshold": False,
        "threshold_adaptation_tau": 100.0
    }

    layer = SNNS(
        input_dim=10,
        hidden_dim=20,
        lif_params=lif_params,
        activation=nn.relu
    )

    # Create sample input
    batch_size, seq_len, input_dim = 2, 5, 10
    x = jax.random.normal(key, (batch_size, seq_len, input_dim))

    # Initialize parameters
    params = layer.init(key, x)

    # Forward pass
    output = layer.apply(params, x)

Dependencies
-----------

- JAX and JAX NumPy for array operations
- Flax for neural network layers
- Logging for operation tracking

See Also
--------

- :doc:`capibara_jax_ssm`
- :doc:`liquid`
- :doc:`meta_bamdp` 
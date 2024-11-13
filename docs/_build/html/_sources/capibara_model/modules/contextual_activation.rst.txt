Contextual Activation Module
===========================

This document provides an overview of the ``contextual_activation.py`` module, which is responsible for managing contextual activation of different components in the CapibaraGPT model.

ScoringContextualActivationModule Class
-------------------------------------

The ``ScoringContextualActivationModule`` class determines component activation based on text relevance scores.

Initialization
^^^^^^^^^^^^^

The class is initialized with the following parameters:

- **ethics_config** (``Dict[str, Any]``): Configuration for ethics module
- **personality_config** (``Dict[str, float]``): Configuration for personality traits
- **enabled** (``bool``, optional): Whether the module is enabled. Default is ``True``

Methods
^^^^^^^

analyze_context(text: str, text_bytes: bytes) -> Dict[str, Any]
---------------------------------------------------------------
Analyzes the context of input text and determines which components should be activated.

Parameters:
    - **text** (``str``): The input text to analyze
    - **text_bytes** (``bytes``): The raw bytes of the input text

Returns:
    A dictionary containing:
        - ethics: Score and activation status for ethics module
        - personality: Score and activation status for personality module 
        - submodels: Scores and activation status for each submodel
        - layers: Scores and activation status for each layer
        - managed_text: The processed text after context management

process_text(text: str) -> Dict[str, Any]
-----------------------------------------

Processes text using TTS and coherence detection.

Parameters:
    - **text** (``str``): The text to process

Returns:
    A dictionary containing:
        - audio: Generated audio output
        - is_coherent: Boolean indicating coherence status

Component Keywords
----------------

The module defines keywords for different components:

Submodels
^^^^^^^^^

.. code-block:: python

    {
        "aleph_tilde": {"logic", "reasoning", "inference"},
        "liquid": {"fluid", "adaptive", "dynamic"},
        "meta_mamdp": {"meta", "learning", "adaptation"},
        "snns_licell": {"spiking", "neural", "network"},
        "spike_ssm": {"spike", "state", "machine"},
        "capibara_jax_ssm": {"jax", "state", "machine"},
        "capibara2": {"next", "generation", "advanced"}
    }

Layers
^^^^^^

.. code-block:: python

    {
        "bitnet_quantizer": {"quantization", "compression", "efficiency"},
        "bitnet": {"binary", "neural", "network"},
        "platonic": {"philosophy", "concepts", "ideals"},
        "game_theory": {"strategy", "decision", "optimization"},
        "self_attention": {"attention", "focus", "relevance"},
        "sparse_capibara": {"sparse", "efficient", "selective"},
        "synthetic_embedding": {"embedding", "representation", "synthetic"}
    }

Integration
----------

The module integrates with several other components:

- Ethics Module
- Personality Manager
- Coherence Manager
- Text-to-Speech
- Various neural network layers and submodels

See Also
--------

- :doc:`ethics_module`
- :doc:`personality_manager`
- :doc:`coherence_manager`

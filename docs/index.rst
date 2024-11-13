.. capibara-gpt documentation master file, created by sphinx-quickstart.
   This file serves as the main entry point for the Capibara-GPT documentation.

Welcome to Capibara-GPT's Documentation!
========================================
Capibara-GPT is a cutting-edge multimodal AI model designed for efficient processing, contextual interaction, and ethical decision-making. This documentation provides detailed information about its architecture, modules, layers, and deployment process.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Main Contents:

   core
   data
   layers
   modules
   optimizations
   sub_models
   utils
 

Capibara Model
--------------

The core of Capibara-GPT's architecture, including the main model components and deployment tools.

Core Components
---------------

.. toctree::
   :maxdepth: 1

   capibara_model/core/config
   capibara_model/core/model
   capibara_model/core/inference
   capibara_model/core/main
   capibara_model/core/optimizer

Sub-Models
----------

.. toctree::
   :maxdepth: 1

   capibara_model/sub_models/liquid
   capibara_model/sub_models/spike_ssm
   capibara_model/sub_models/aleph_Tilde
   capibara_model/sub_models/capibara_byte
   capibara_model/sub_models/capibara_jax_ssm
   capibara_model/sub_models/capibara2
   capibara_model/sub_models/capibara_jax_ssm
   capibara_model/sub_models/deep_dialog
   capibara_model/sub_models/snns_LiCell
   capibara_model/sub_models/meta_bamdp


Modules
-------

.. toctree::
   :maxdepth: 1

   capibara_model/modules/contextual_activation
   capibara_model/modules/conversation_manager
   capibara_model/modules/ethics_module
   capibara_model/modules/personality_module
   capibara_model/modules/capibara_tts
   capibara_model/modules/coherence_module

Optimizations
-------------

.. toctree::
   :maxdepth: 1

   capibara_model/optimize/src/lib.rs
   

Model Conversion
----------------

.. toctree::
   :maxdepth: 1

   capibara_model/ONNX_conversion/build_model
   capibara_model/ONNX_conversion/optimize_model
   capibara_model/ONNX_conversion/convert_to_tflite

Utils
-----

.. toctree::
   :maxdepth: 1

   capibara_model/utils/activations
   capibara_model/utils/data_processing
   capibara_model/utils/generate_response
   capibara_model/utils/language_utils
   

Configuration
-------------

.. toctree::
   :maxdepth: 1

   config/base
   config/development
   config/production
   config/testing
   config/tpu_config
   config/param_configs.yaml
   config/config.yaml

Data
----

.. toctree::
   :maxdepth: 1

   capibara_model/data/data_loader
   capibara_model/data/dataset

Layers
------

.. toctree::
   :maxdepth: 1

   capibara_model/layers/bitnet_quantizer
   capibara_model/layers/bitnet
   capibara_model/layers/synthetic_embedding
   capibara_model/layers/self_attention
   capibara_model/layers/games_theory
   capibara_model/layers/platonic
   capibara_model/layers/mixture_of_rookies
   capibara_model/layers/sparse_capibara
   capibara_model/layers/MetaLa
   capibara_model/layers/quineana


Testing
-------

.. toctree::
   :maxdepth: 1

   tests/test_model
   tests/test_layer
   tests/conftest
   tests/run_all_tests

API Reference
-------------

.. toctree::
   :maxdepth: 1

   



Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

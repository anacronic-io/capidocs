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

   capibara_model/core/model
   capibara_model/core/inference
   capibara_model/core/config
   capibara_model/data/dataset
   capibara_model/deployment/docker_manager

Capibara Model
--------------

The core of Capibara-GPT's architecture, including the main model components and deployment tools.

Core Components
~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   capibara_model/core/config
   capibara_model/core/model
   capibara_model/core/inference

Sub-Models
~~~~~~~~~

.. toctree::
   :maxdepth: 1

   capibara_model/sub_models/liquid
   capibara_model/sub_models/spike_ssm
   capibara_model/sub_models/aleph_Tilde
   capibara_model/sub_models/capibara_byte
   capibara_model/sub_models/capibara_jax_ssm

Deployment
~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   capibara_model/deployment/deployer
   capibara_model/deployment/docker_manager

Modules
~~~~~~~

.. toctree::
   :maxdepth: 1

   capibara_model/modules/contextual_activation
   capibara_model/modules/conversation_manager
   capibara_model/modules/ethics_module
   capibara_model/modules/personality_module
   capibara_model/modules/aleph_module

Optimizations
~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   capibara_model/optimizations/rust_optimization
   capibara_model/optimizations/jax_optimization
   capibara_model/optimizations/memory_optimization
   capibara_model/ONNX_conversion/optimize_model

Model Conversion
~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   capibara_model/ONNX_conversion/build_model
   capibara_model/ONNX_conversion/optimize_model
   capibara_model/ONNX_conversion/convert_to_tflite

Utils
~~~~~

.. toctree::
   :maxdepth: 1

   capibara_model/utils/activations
   capibara_model/utils/data_processing
   capibara_model/utils/generate_response
   capibara_model/utils/language_utils
   capibara_model/utils/logging

Configuration
-------------

.. toctree::
   :maxdepth: 1

   config/base
   config/development
   config/production
   config/testing
   config/tpu_config

Data
----

.. toctree::
   :maxdepth: 1

   capibara_model/data/data_loader
   capibara_model/data/dataset
   capibara_model/data/preprocessing

Layers
------

.. toctree::
   :maxdepth: 1

   capibara_model/layers/bitnet_quantizer
   capibara_model/layers/bitnet
   capibara_model/layers/mamba2
   capibara_model/layers/mamba_byte
   capibara_model/layers/synthetic_embedding
   capibara_model/layers/sparse_mamba
   capibara_model/layers/self_attention
   capibara_model/layers/liquid
   capibara_model/layers/meta_bamdp
   capibara_model/layers/games_theory
   capibara_model/layers/platonic
   capibara_model/layers/snns_layer
   capibara_model/layers/capibara_ssm

Testing
-------

.. toctree::
   :maxdepth: 1

   tests/unit_tests
   tests/integration_tests
   tests/performance_tests
   tests/conftest

API Reference
------------

.. toctree::
   :maxdepth: 1

   api/core
   api/layers
   api/modules
   api/utils

Contributing
-----------

.. toctree::
   :maxdepth: 1

   contributing/guidelines
   contributing/development_setup
   contributing/code_style

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

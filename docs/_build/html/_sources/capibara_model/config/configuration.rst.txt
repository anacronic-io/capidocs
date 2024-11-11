Configuration
============

Overview
--------
The configuration system for CapibaraGPT provides a flexible way to configure model parameters, training settings, and inference options.

Core Configuration
----------------

Model Settings
~~~~~~~~~~~~~

.. code-block:: yaml

    model:
      name: "capibara"
      size: "34B"
      precision: "bfloat16"
      max_sequence_length: 8192
      vocab_size: 32000
      hidden_size: 4096
      num_attention_heads: 32
      num_hidden_layers: 32
      intermediate_size: 11008
      hidden_act: "silu"
      rotary_pct: 0.25
      rotary_emb_base: 10000
      layer_norm_epsilon: 1e-5
      use_cache: true
      tie_word_embeddings: false
      pad_token_id: 0
      bos_token_id: 1
      eos_token_id: 2

Training Settings
~~~~~~~~~~~~~~~

.. code-block:: yaml

    training:
      optimizer:
        name: "adamw"
        lr: 2.0e-5
        weight_decay: 0.1
        beta1: 0.9
        beta2: 0.95
        eps: 1.0e-8
        
      scheduler:
        name: "cosine"
        warmup_steps: 2000
        
      batch_size: 32
      gradient_accumulation_steps: 8
      max_steps: 50000
      save_steps: 1000
      eval_steps: 500
      logging_steps: 100

Inference Settings
~~~~~~~~~~~~~~~~

.. code-block:: yaml

    inference:
      temperature: 0.7
      top_p: 0.95
      top_k: 40
      repetition_penalty: 1.1
      max_new_tokens: 512
      do_sample: true
      num_beams: 1
      early_stopping: true

Data Settings
~~~~~~~~~~~~

.. code-block:: yaml

    data:
      train_file: "data/train.jsonl"
      validation_file: "data/validation.jsonl" 
      test_file: "data/test.jsonl"
      max_train_samples: null
      max_eval_samples: null
      preprocessing_num_workers: 8
      overwrite_cache: false

System Settings
~~~~~~~~~~~~~

.. code-block:: yaml

    system:
      seed: 42
      dtype: "bfloat16"
      device: "cuda"
      distributed_type: "multi-gpu"
      mixed_precision: true
      gradient_checkpointing: true
      torch_compile: true
      flash_attention: true

Usage
-----

Loading Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from capibara_model.core.config import CapibaraConfig
    
    config = CapibaraConfig.from_yaml("path/to/config.yaml")

Modifying Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    config.model.max_sequence_length = 4096
    config.training.batch_size = 16
    
    # Save updated config
    config.save_yaml("new_config.yaml")

See Also
--------
- :doc:`../core/config`: Configuration system documentation
- :doc:`../utils/yaml`: YAML utilities documentation
- :doc:`../training/trainer`: Trainer documentation

Notes
-----
- All parameters can be overridden via command line arguments
- Default values are optimized for 34B parameter model
- Memory settings should be adjusted based on available hardware
- Consider using gradient checkpointing for large models
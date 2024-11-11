Test Utilities Module
====================

.. module:: tests.test_utils
   :synopsis: Test suite for the CapibaraModel utilities and core functionality.

This module contains unit tests for the CapibaraModel's utilities, including model creation, 
forward passes, response generation, and various utility functions.

Test Classes
-----------

TestCapibaraModel
~~~~~~~~~~~~~~~~

.. autoclass:: TestCapibaraModel
   :members:
   :private-members:
   :special-members: __init__

   Main test class for the CapibaraModel functionality.

Test Methods
-----------

Model Tests
~~~~~~~~~~

.. method:: test_model_creation()

   Tests the proper instantiation of the CapibaraModel.

.. method:: test_forward_pass()

   Validates the model's forward pass with random input data.

.. method:: test_generate_response()

   Tests the model's response generation capabilities with sample conversation history.

Activation Function Tests
~~~~~~~~~~~~~~~~~~~~~~~

.. method:: test_gelu()

   Tests the Gaussian Error Linear Unit (GELU) activation function.

.. method:: test_swish()

   Tests the Swish activation function implementation.

Architecture Tests
~~~~~~~~~~~~~~~~

.. method:: test_positional_encoding()

   Validates the positional encoding implementation.

.. method:: test_create_masks()

   Tests the creation of source and target masks for attention mechanisms.

Language Processing Tests
~~~~~~~~~~~~~~~~~~~~~~~

.. method:: test_detect_language()

   Tests the language detection functionality.

.. method:: test_translate_text()

   Tests the text translation capabilities between languages.

Configuration
------------

The test suite uses the following configuration:

.. code-block:: python

   config = CapibaraConfig(
       d_model=512,
       d_state=256,
       d_conv=128,
       expand=2,
       base_model_name='gpt2',
       translation_model='facebook/m2m100_418M',
       get_active_layers=lambda: ['platonic', 'game_theory', 'ethics'],
       get_layer_config=lambda layer_name: {},
       personality={},
       context_window_size=10,
       max_length=50,
       vocab_size=1000
   )

See Also
--------

- :doc:`../core/config`
- :doc:`../core/model`
- :doc:`../utils/data_processing`
- :doc:`../utils/generate_response`
- :doc:`../utils/language_utils` 
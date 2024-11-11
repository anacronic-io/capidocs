Coherence Detector Module
========================

This module provides functionality for detecting coherence in text using JAX/Flax and pre-trained models.

CoherenceDetector Class
----------------------

.. py:class:: CoherenceDetector(model_name: str)

   A class for detecting coherence in text using a pre-trained model with JAX/Flax.

   :param model_name: The name of the pre-trained model to use
   :type model_name: str
   :raises Exception: If the model or tokenizer fails to load

   .. py:attribute:: tokenizer
      
      Tokenizer for the pre-trained model

   .. py:attribute:: model
      
      Pre-trained model for next sentence prediction

   .. py:attribute:: threshold
      
      Threshold for determining coherence

   .. py:attribute:: device
      
      JAX device to run computations on

Methods
^^^^^^^

.. py:method:: detect_coherence(context: str, response: str) -> bool

   Determines if the response is coherent with the context.

   :param context: The context text
   :type context: str
   :param response: The response text to evaluate
   :type response: str
   :return: True if the response is coherent with the context, False otherwise
   :rtype: bool
   :raises Exception: If an error occurs during coherence detection

Example Usage
------------

.. code-block:: python

   from capibara_model.modules.coherence_detector import CoherenceDetector

   # Initialize the detector
   detector = CoherenceDetector("bert-base-uncased")

   # Check coherence
   context = "The weather is sunny today."
   response = "I should bring my sunglasses."
   is_coherent = detector.detect_coherence(context, response)

Environment Variables
-------------------

The module uses the following environment variables:

- ``CAPIBARA_COHERENCE_THRESHOLD``: Threshold value for coherence detection (default: 0.5)

Dependencies
-----------

- jax
- jax.numpy
- transformers (FlaxBertForNextSentencePrediction, AutoTokenizer)
- logging
- os

Implementation Details
--------------------

The coherence detection is implemented using:

1. A pre-trained BERT model for next sentence prediction
2. JAX/Flax for efficient computation
3. Tokenization of input texts
4. Softmax scoring of model outputs
5. Threshold-based decision making

The model computes a coherence score between 0 and 1, where scores above the threshold 
(configurable via environment variable) indicate coherent responses.

See Also
--------

- :doc:`contextual_activation`
- :doc:`conversation_manager`
- :doc:`ethics_module`

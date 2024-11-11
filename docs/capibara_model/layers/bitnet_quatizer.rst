BitNet Quantizer
================

.. automodule:: layers.bitnet_quantizer
   :members:
   :undoc-members:
   :show-inheritance:

The BitNet Quantizer module is designed to facilitate the quantization of neural network layers, specifically tailored for the BitNet architecture. Quantization is a crucial technique in deep learning that reduces the precision of the weights and activations, leading to reduced model size and faster inference times without significantly compromising accuracy.

Overview
--------

The BitNet Quantizer provides tools to convert floating-point operations into lower precision, such as 8-bit integers. This is particularly useful for deploying models on resource-constrained devices where computational efficiency and memory usage are critical.

Key Features
^^^^^^^^^^^^

- **Precision Reduction**: Converts high-precision weights and activations to lower precision, optimizing model performance.
- **Compatibility**: Designed to integrate seamlessly with the BitNet architecture, ensuring minimal disruption to existing workflows.
- **Configurable**: Allows customization of quantization parameters to balance between model size and accuracy.

Classes and Methods
-------------------

.. autoclass:: layers.bitnet_quantizer.BitNetQuantizer
   :members:
   :undoc-members:
   :show-inheritance:

   The `BitNetQuantizer` class is the core component of this module, providing methods to apply quantization to neural network layers.

   .. automethod:: __init__

      Initializes the quantizer with specified parameters, setting up the necessary configurations for quantization.

   .. automethod:: quantize_weights

      Applies quantization to the weights of a given layer, reducing their precision.

   .. automethod:: quantize_activations

      Applies quantization to the activations of a given layer, optimizing runtime performance.

   .. automethod:: dequantize

      Reverts quantized weights and activations back to their original precision, if needed.

Usage Example
-------------

Below is an example demonstrating how to utilize the `BitNetQuantizer`:

.. code-block:: python

   from layers.bitnet_quantizer import BitNetQuantizer
   import torch.nn as nn

   # Initialize a sample neural network layer
   layer = nn.Linear(512, 256)

   # Initialize the BitNetQuantizer
   quantizer = BitNetQuantizer(precision=8)

   # Quantize the weights of the layer
   quantized_weights = quantizer.quantize_weights(layer.weight)

   # Quantize the activations of the layer
   quantized_activations = quantizer.quantize_activations(layer(torch.randn(1, 512)))

   # Dequantize if needed
   original_weights = quantizer.dequantize(quantized_weights)

   print("Quantization complete.")

Notes
-----

- Quantization can significantly reduce model size and improve inference speed, but it may also lead to a slight decrease in accuracy. It is important to test and validate the quantized model thoroughly.
- The `BitNetQuantizer` is designed to be flexible, allowing users to adjust the precision level according to their specific needs and constraints.
- Ensure that the quantization parameters are compatible with the target deployment environment to achieve optimal results.
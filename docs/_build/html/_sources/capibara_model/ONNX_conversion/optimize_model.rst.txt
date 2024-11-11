Model Optimization
================

The optimization module provides tools for quantizing and optimizing ONNX models to improve performance and reduce size.

optimize_model
------------

.. automodule:: capibara_model.ONNX_conversion.optimize_model
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example:

.. code-block:: python

   from capibara_model.ONNX_conversion.optimize_model import optimize_onnx_model
   
   # Quantize ONNX model to INT8
   optimize_onnx_model(
       input_model_path="model.onnx",
       output_model_path="model_quantized.onnx"
   )
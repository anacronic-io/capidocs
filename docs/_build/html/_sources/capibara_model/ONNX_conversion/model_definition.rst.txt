TFLite Conversion
================

The TFLite conversion module enables converting ONNX models to TensorFlow Lite format for mobile and edge deployment.

convert_to_tflite
---------------

.. automodule:: capibara_model.ONNX_conversion.convert_to_tflite
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example:

.. code-block:: python

   from capibara_model.ONNX_conversion.convert_to_tflite import convert_onnx_to_tflite
   
   # Convert ONNX model to TFLite
   convert_onnx_to_tflite(
       input_model_path="model_quantized.onnx",
       output_tflite_path="model.tflite"
   )
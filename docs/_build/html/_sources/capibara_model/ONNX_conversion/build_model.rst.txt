ONNX Conversion
================

The ONNX conversion module allows exporting the Capibara model to ONNX format for deployment.

build_model
----------

.. automodule:: capibara_model.ONNX_conversion.build_model
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example:

.. code-block:: python

   from capibara_model.ONNX_conversion.build_model import build_and_save_onnx_model
   
   # Export model to ONNX
   build_and_save_onnx_model("my_model.onnx")
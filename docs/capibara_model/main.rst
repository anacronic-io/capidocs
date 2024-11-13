Main Module
===========

The main module is the entry point for the CapibaraModel application. It is located in `capibara_model/main.py`.

Main Function
-------------

.. autofunction:: capibara_model.main.main

The `main` function orchestrates the entire execution of the CapibaraModel application:

1. Parses command-line arguments using `parse_arguments`. It allows specifying the model name and configuration file path.

2. Sets up logging using `setup_logging`. It configures logging based on the provided log level and logging configuration file.

3. Loads the configuration from the specified YAML file using `load_config`.

4. Initializes the resource manager using `resource_manager`. It ensures proper initialization and cleanup of model resources.

5. Trains the model by calling `train_model` and passing the loaded configuration.

6. Evaluates the trained model by calling `evaluate_model` and passing the loaded configuration.

7. Deploys the model by calling `deploy_model` and passing the loaded configuration.

8. Handles any exceptions that occur during execution and logs them appropriately.

Utility Functions
-----------------

.. autofunction:: capibara_model.main.parse_arguments
.. autofunction:: capibara_model.main.setup_logging  
.. autofunction:: capibara_model.main.load_config
.. autofunction:: capibara_model.main.resource_manager
Test Configuration Module
========================

.. module:: tests.conftest
   :synopsis: Test configuration and fixtures for the CapibaraModel test suite.

This module provides the test configuration, fixtures and setup for testing the CapibaraModel.

Functions
---------

initialize_tests
~~~~~~~~~~~~~~

.. function:: initialize_tests() -> None

   Initialize the necessary configuration for running all tests.
   
   Sets up:
   
   - Random seeds for reproducibility
   - Environment variables 
   - JAX platform configuration
   - Logging configuration
   - Test data directories

   :raises RuntimeError: If critical initialization steps fail

Fixtures
--------

setup_tests
~~~~~~~~~~

.. fixture:: setup_tests() -> None
   :scope: session
   :autouse: True

   Session-wide test setup fixture that runs once at the beginning of the test session.

test_config 
~~~~~~~~~~

.. fixture:: test_config() -> CapibaraConfig
   :scope: session

   Provides a test configuration for the CapibaraModel.

   :returns: A configuration object with test settings
   :rtype: CapibaraConfig

capibara_model
~~~~~~~~~~~~

.. fixture:: capibara_model(test_config: CapibaraConfig) -> CapibaraModel  
   :scope: function

   Provides a fresh instance of the CapibaraModel for each test.

   :param test_config: The test configuration fixture
   :returns: A new model instance
   :rtype: CapibaraModel

rng_key
~~~~~~~

.. fixture:: rng_key() -> jnp.ndarray
   :scope: function

   Provides a fresh PRNG key for each test.

   :returns: A JAX PRNG key
   :rtype: jnp.ndarray

sample_input
~~~~~~~~~~~

.. fixture:: sample_input(rng_key: jnp.ndarray, test_config: CapibaraConfig) -> jnp.ndarray
   :scope: function

   Provides a sample input tensor for testing.

   :param rng_key: The random key fixture
   :param test_config: The test configuration fixture
   :returns: A random input tensor
   :rtype: jnp.ndarray

model_params
~~~~~~~~~~

.. fixture:: model_params(capibara_model: CapibaraModel, sample_input: jnp.ndarray, rng_key: jnp.ndarray) -> Dict[str, Any]
   :scope: function

   Provides initialized model parameters.

   :param capibara_model: The model fixture
   :param sample_input: The sample input fixture 
   :param rng_key: The random key fixture
   :returns: The initialized model parameters
   :rtype: Dict[str, Any]

attention_mask
~~~~~~~~~~~~

.. fixture:: attention_mask(test_config: CapibaraConfig) -> jnp.ndarray
   :scope: function

   Provides an attention mask for testing.

   :param test_config: The test configuration fixture
   :returns: An attention mask tensor
   :rtype: jnp.ndarray

test_data_dir
~~~~~~~~~~~~

.. fixture:: test_data_dir() -> Path
   :scope: session

   Provides the path to the test data directory.

   :returns: Path to the test data directory
   :rtype: Path

Pytest Hooks
-----------

pytest_configure
~~~~~~~~~~~~~~

.. function:: pytest_configure(config)

   Allows plugins and conftest files to perform initial configuration.

pytest_sessionfinish  
~~~~~~~~~~~~~~~~~~

.. function:: pytest_sessionfinish(session, exitstatus)

   Called after whole test run finished.

pytest_runtest_setup
~~~~~~~~~~~~~~~~~~

.. function:: pytest_runtest_setup(item)

   Called before running a test item.

pytest_runtest_teardown
~~~~~~~~~~~~~~~~~~~~~

.. function:: pytest_runtest_teardown(item, nextitem)

   Called after running a test item.

See Also
--------

- :doc:`../core/config`
- :doc:`../model`
- :doc:`../utils/logging` 
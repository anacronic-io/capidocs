Test Runner Module
=================

.. module:: tests.run_all_tests
   :synopsis: Main test runner for the CapibaraModel test suite.

Este módulo proporciona la funcionalidad principal para ejecutar todas las pruebas del modelo CapibaraGPT.

Functions
---------

generate_mock_data
~~~~~~~~~~~~~~~~

.. function:: generate_mock_data(key, num_samples, sequence_length, vocab_size)

   Genera datos simulados para pruebas.

   :param key: Clave aleatoria JAX para generar datos
   :type key: jax.random.PRNGKey
   :param num_samples: Número de muestras de datos simulados a generar
   :type num_samples: int
   :param sequence_length: Longitud de cada secuencia de datos simulados
   :type sequence_length: int
   :param vocab_size: Tamaño del vocabulario para generar texto aleatorio
   :type vocab_size: int
   :return: Lista de diccionarios conteniendo datos 'text' para pruebas
   :rtype: list

create_test_config
~~~~~~~~~~~~~~~~

.. function:: create_test_config()

   Crea una configuración de prueba.

   :return: Objeto de configuración con valores predefinidos
   :rtype: CapibaraConfig

load_tests
~~~~~~~~~

.. function:: load_tests(loader, standard_tests, pattern)

   Carga todas las pruebas desde archivos en la carpeta tests.

   :param loader: Objeto cargador de pruebas
   :type loader: unittest.TestLoader
   :param standard_tests: Suite de pruebas estándar
   :type standard_tests: unittest.TestSuite
   :param pattern: Patrón para coincidir con archivos de prueba
   :type pattern: str
   :return: Suite de pruebas actualizada con las pruebas cargadas
   :rtype: unittest.TestSuite

Classes
-------

JAXTestRunner
~~~~~~~~~~~

.. class:: JAXTestRunner

   Clase ejecutora de pruebas basadas en JAX.

   .. method:: run()
      :staticmethod:

      Ejecuta todas las pruebas usando el ejecutor de pruebas JAX.
      
      Este método:
      
      - Configura el entorno de pruebas
      - Crea datos simulados
      - Crea un dataset y dataloader de prueba
      - Carga todos los archivos de prueba
      - Ejecuta las pruebas usando unittest

Usage Example
------------

.. code-block:: python

   if __name__ == "__main__":
       # Configurar JAX para usar CPU, GPU o TPU según sea necesario
       # jax.config.update('jax_platform_name', 'cpu')  # Para CPU
       # jax.config.update('jax_platform_name', 'gpu')  # Para GPU
       # jax.config.update('jax_platform_name', 'tpu')  # Para TPU

       JAXTestRunner.run()

See Also
--------

- :doc:`test_utils`
- :doc:`conftest`
- :doc:`../core/config`
- :doc:`../data/dataset`

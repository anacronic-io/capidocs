Multilingual Dataset
==================

Overview
--------
The Multilingual Dataset module provides functionality for handling multilingual text data in the CapibaraGPT model. It includes language detection, translation, and tokenization capabilities.

Core Features
------------
- Language detection and validation
- Automatic translation of unsupported languages
- Efficient caching of translations
- Dataset splitting for train/val/test
- Integration with TensorFlow Datasets
- Google Cloud Storage support
- Memory-efficient data loading

MultilingualDataset Class
------------------------

.. autoclass:: capibara_model.data.dataset.MultilingualDataset
   :members:
   :undoc-members:
   :special-members: __init__

Key Components
~~~~~~~~~~~~~

Data Loading
^^^^^^^^^^^
- TensorFlow Datasets integration
- Google Cloud Storage support
- Local file loading
- Memory-efficient data streaming

Language Processing
^^^^^^^^^^^^^^^^^
- Automatic language detection
- Translation of unsupported languages
- Language distribution analysis
- Caching of translations

Tokenization
^^^^^^^^^^^
- Integration with HuggingFace tokenizers
- Configurable maximum sequence length
- Batch tokenization support
- Special token handling

Dataset Operations
^^^^^^^^^^^^^^^^
- Train/validation/test splitting
- Shuffling and sampling
- Length-based filtering
- Data augmentation

Dependencies
-----------
- transformers
- langdetect
- googletrans
- tensorflow_datasets
- google-cloud-storage
- python-dotenv

Configuration
------------
The module uses environment variables for configuration:

- ``GOOGLE_APPLICATION_CREDENTIALS``: GCS authentication
- ``MAX_SEQUENCE_LENGTH``: Maximum tokenization length
- ``CACHE_DIR``: Directory for caching translations
- ``SUPPORTED_LANGUAGES``: List of supported language codes

Usage Examples
-------------

Basic Usage
^^^^^^^^^^
.. code-block:: python

    from capibara_model.data import MultilingualDataset
    
    data = [{"text": "Hello world"}, {"text": "Hola mundo"}]
    langs = ["en", "es"]
    
    dataset = MultilingualDataset(data, langs)
    train, val, test = dataset.split_dataset()

Advanced Usage
^^^^^^^^^^^^^
.. code-block:: python

    dataset = MultilingualDataset.from_tfds("wmt14_translate/es-en")
    dataset.load_from_gcs("gs://bucket/path")
    
    stats = dataset.get_language_distribution()
    filtered = dataset.filter_by_length(max_length=512)

See Also
--------
- :doc:`data_loader`: Data loading utilities
- :doc:`tokenizer`: Tokenization documentation
- :doc:`../utils/gcs`: Google Cloud Storage utilities

Notes
-----
- Ensure proper environment variables are set
- Monitor translation API usage limits
- Consider caching for large datasets
- Test language detection accuracy
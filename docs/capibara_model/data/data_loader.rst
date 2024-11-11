Data Loader
==========

Overview
--------
The data loader module provides efficient data loading and preprocessing functionality for the CapibaraModel. It handles batching, shuffling, and preprocessing of training and inference data.

Core Features
------------
- Efficient batch processing with TensorFlow datasets
- Support for multilingual data
- Dynamic batch size handling
- Customizable data preprocessing
- Memory-efficient data loading
- Integration with JAX/NumPy arrays

Architecture
-----------

CapibaraDataLoader
~~~~~~~~~~~~~~~~

.. autoclass:: capibara_model.data.data_loader.CapibaraDataLoader
   :members:
   :undoc-members:
   :special-members: __init__

Key Components
~~~~~~~~~~~~

1. **Data Loading**
   - TensorFlow datasets integration
   - Custom dataset support
   - Error handling and logging

2. **Batch Processing**
   - Dynamic batch creation
   - Memory-efficient generators
   - Customizable collation

3. **Preprocessing Pipeline**
   - Tokenization
   - Attention mask generation
   - Label processing

Implementation Details
--------------------

Data Generator
~~~~~~~~~~~~

.. code-block:: python

    def _data_generator(self):
        """
        Generator for iterating over data in batches.
        
        Yields:
            Dict containing batched tensors:
                - input_ids: Token IDs
                - attention_mask: Attention masks
                - labels: Target labels
        """

Batch Collation
~~~~~~~~~~~~~

.. code-block:: python

    def _collate_fn(self, batch: List[Dict]) -> Dict[str, jnp.ndarray]:
        """
        Custom collation for batch creation.
        
        Args:
            batch: List of individual examples
            
        Returns:
            Dictionary of batched arrays
        """

Usage Examples
------------

Basic Usage
~~~~~~~~~

.. code-block:: python

    from capibara_model.data.data_loader import CapibaraDataLoader
    from capibara_model.core.config import CapibaraConfig
    from transformers import AutoTokenizer

    # Initialize components
    config = CapibaraConfig()
    tokenizer = AutoTokenizer.from_pretrained("nous-capybara")

    # Create data loader
    loader = CapibaraDataLoader(
        config=config,
        tokenizer=tokenizer,
        dataset_name="wikitext",
        batch_size=32,
        shuffle=True
    )

    # Iterate over batches
    for batch in loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

Configuration Options
------------------

.. code-block:: python

    class DataLoaderConfig:
        """Configuration for data loader."""
        
        batch_size: int = 32
        shuffle: bool = True
        num_workers: int = 4
        max_length: int = 2048
        dataset_name: str = "wikitext"
        cache_dir: Optional[str] = None

Performance Optimization
---------------------

Memory Management
~~~~~~~~~~~~~~
- Efficient batch generation
- Prefetching support
- Dynamic memory allocation

Throughput Optimization
~~~~~~~~~~~~~~~~~~~
- Parallel data loading
- Efficient preprocessing
- Caching mechanisms

Dependencies
----------
- tensorflow
- jax
- numpy
- transformers
- tensorflow_datasets

See Also
--------
- :doc:`dataset`: Dataset documentation
- :doc:`../core/config`: Configuration system
- :doc:`../utils/tokenizer`: Tokenizer documentation

Notes
-----
- Ensure proper dataset configuration before initialization
- Monitor memory usage with large batch sizes
- Consider using caching for repeated data access
- Adjust num_workers based on available CPU cores
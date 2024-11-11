Language Utilities Module
=======================

.. module:: language_utils
   :synopsis: Language processing utilities for the CapibaraModel model.

This module provides utility functions for language processing in the CapibaraModel model, including tokenization, detokenization, language detection, and other language-related tasks.

Functions
--------

tokenize
~~~~~~~

.. function:: tokenize(text: str) -> List[str]

   Tokenizes the given text into a list of tokens.

   :param text: Input text to tokenize
   :type text: str
   :return: List of tokens
   :rtype: List[str]

detokenize
~~~~~~~~~

.. function:: detokenize(tokens: List[str]) -> str

   Detokenizes a list of tokens back into text.

   :param tokens: List of tokens to combine
   :type tokens: List[str]
   :return: Combined text
   :rtype: str

normalize_text
~~~~~~~~~~~~

.. function:: normalize_text(text: str) -> str

   Normalizes text by converting to lowercase and removing extra whitespace.

   :param text: Input text to normalize
   :type text: str
   :return: Normalized text
   :rtype: str

detect_language
~~~~~~~~~~~~~

.. function:: detect_language(text: str) -> str

   Detects the language of the given text.

   :param text: Input text
   :type text: str
   :return: Language code (e.g., 'en', 'es', 'fr')
   :rtype: str
   :raises: Logs warning if text is empty
   :raises: Logs error if language detection fails

is_valid_language
~~~~~~~~~~~~~~~

.. function:: is_valid_language(text: str, allowed_languages: List[str]) -> bool

   Checks if text language is among allowed languages.

   :param text: Input text
   :type text: str
   :param allowed_languages: List of allowed language codes
   :type allowed_languages: List[str]
   :return: True if language is valid, False otherwise
   :rtype: bool

translate_text
~~~~~~~~~~~~

.. function:: translate_text(text: str, source_lang: str, target_lang: str) -> str

   Translates text between languages.

   :param text: Text to translate
   :type text: str
   :param source_lang: Source language code
   :type source_lang: str
   :param target_lang: Target language code
   :type target_lang: str
   :return: Translated text
   :rtype: str
   :note: Currently a placeholder implementation

Dependencies
-----------

- re
- logging
- typing
- langdetect

Example Usage
------------

.. code-block:: python

    from capibara_model.utils.language_utils import (
        tokenize,
        normalize_text,
        detect_language
    )

    # Tokenize text
    text = "Hello, world!"
    tokens = tokenize(text)
    # ['Hello', ',', 'world', '!']

    # Normalize text
    normalized = normalize_text("Hello   World  ")
    # "hello world"

    # Detect language
    lang = detect_language("Bonjour le monde")
    # "fr"

Configuration
------------

The module uses the following configuration:

- DetectorFactory.seed = 0 (ensures consistent language detection)
- Logging configured for warnings and errors

See Also
--------

- :doc:`data_processing`
- :doc:`tokenization`
- :doc:`generate_response` 
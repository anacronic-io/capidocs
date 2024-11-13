Conversation Manager Module
===========================

This document provides an overview of the `conversation_manager.py` module, which is responsible for managing conversation coherence and ethical responses in the CapibaraENT model. The module integrates various components such as ethics checking, personality management, and text-to-speech synthesis.

CoherenceManager Class
----------------------

The `CoherenceManager` class handles the coherence of responses in conversations, ensuring that the dialogue remains on-topic and ethically appropriate.

Initialization

The class is initialized with the following parameters:

- **ethics_module** (`EthicsModule`): Module for filtering unethical words.
- **personality_manager** (`PersonalityManager`): Manages personality adjustments for responses.

The class also initializes a text-to-speech model:

```python
self.tts_model = CapibaraTextToSpeech(
    fastspeech_model_path="path/to/fastspeech/model",
    hifigan_model_path="path/to/hifigan/model"
)
```

Methods

#`get_coherence_response(response, context)`

Selects a random response to use when coherence is lost.

- **response** (`str`): The current response text.
- **context** (`str`): The context of the conversation.

Returns a string that is either the original response or an adjusted, ethical response if coherence is lost.

`add_coherence_response(response)`

Adds a new coherence response to the list.

- **response** (`str`): The new coherence response to add.

`clear_coherence_responses()`

Clears all coherence responses from the list.

`check_ethics(response)`

Checks if a response is ethical.

- **response** (`str`): The response text to check.

Returns a boolean indicating whether the response is ethical.

`initialize()`

Class method to initialize any global settings for `CoherenceManager`.

Logging
-------

The module uses Python's `logging` library to log errors and other information. Ensure that the logging configuration is set up in your application to capture these logs.

```python
import logging

logger = logging.getLogger(__name__)
```

This document should be updated regularly to reflect any changes in the conversation management logic or the addition of new features.


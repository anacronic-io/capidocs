Capibara Text-to-Speech Module
=============================

This document provides an overview of the ``capibara_tts.py`` module, which implements text-to-speech functionality for the CapibaraGPT model using FastSpeech2 and HiFi-GAN.

Environment Variables
-------------------

The module uses the following environment variables:

.. code-block:: python

    FASTSPEECH_MODEL_PATH: Path to FastSpeech2 model
    HIFIGAN_MODEL_PATH: Path to HiFi-GAN model
    CAPIBARA_TTS_SAMPLE_RATE: Sample rate (default: 22050)
    CAPIBARA_TTS_HOST: WebSocket host (default: "localhost")
    CAPIBARA_TTS_PORT: WebSocket port (default: 8765)
    CAPIBARA_TTS_CERT_FILE: SSL certificate file path
    CAPIBARA_TTS_KEY_FILE: SSL key file path

CapibaraTextToSpeech Class
-------------------------

Main class for text-to-speech conversion using FastSpeech2 and HiFi-GAN models.

Initialization
^^^^^^^^^^^^^

.. code-block:: python

    def __init__(self, fastspeech_model_path, hifigan_model_path, sample_rate=22050):
        """
        Initialize TTS with FastSpeech2 and HiFi-GAN models.

        Args:
            fastspeech_model_path (str): Path to FastSpeech2 model
            hifigan_model_path (str): Path to HiFi-GAN model
            sample_rate (int): Audio sample rate (default: 22050)
        """

Methods
^^^^^^^

text_to_spectrogram(text: str) -> np.ndarray
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Converts text to mel spectrogram using FastSpeech2.

Parameters:
    - **text** (``str``): Input text to convert

Returns:
    ``np.ndarray``: Generated mel spectrogram

spectrogram_to_audio(spectrogram: np.ndarray) -> np.ndarray
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Converts mel spectrogram to audio using HiFi-GAN.

Parameters:
    - **spectrogram** (``np.ndarray``): Input mel spectrogram

Returns:
    ``np.ndarray``: Generated audio waveform

handle_connection(websocket, path)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Handles WebSocket connections for real-time TTS.

Parameters:
    - **websocket**: WebSocket connection object
    - **path**: Connection path

start_websocket_server(host="localhost", port=8765, ssl_context=None)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Starts the WebSocket server for TTS requests.

Parameters:
    - **host** (``str``): Server host (default: "localhost")
    - **port** (``int``): Server port (default: 8765)
    - **ssl_context**: SSL context for secure connections

synthesize(text: str) -> bytes
~~~~~~~~~~~~~~~~~~~~~~~~~~

Generates audio from text using pyttsx3 as fallback.

Parameters:
    - **text** (``str``): Text to convert to speech

Returns:
    ``bytes``: Generated audio data

generar_audio(texto: str) -> bytes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapper method for synthesize().

Parameters:
    - **texto** (``str``): Text to convert to speech

Returns:
    ``bytes``: Generated audio data

WebSocket Protocol
----------------

The module implements a WebSocket server that accepts JSON messages with the following format:

Request:

.. code-block:: json

    {
        "text": "Text to convert to speech"
    }

Response:

.. code-block:: json

    {
        "audio": [...],  // Audio data as array
        "sample_rate": 22050
    }

Error Response:

.. code-block:: json

    {
        "error": "Error message"
    }

SSL Configuration
---------------

For secure WebSocket connections (WSS), provide SSL certificate and key files through environment variables:

.. code-block:: bash

    CAPIBARA_TTS_CERT_FILE=path/to/cert.pem
    CAPIBARA_TTS_KEY_FILE=path/to/key.pem

Example Usage
-----------

Basic usage:

.. code-block:: python

    from capibara_model.modules.capibara_tts import CapibaraTextToSpeech

    # Initialize TTS
    tts = CapibaraTextToSpeech(
        fastspeech_model_path="path/to/fastspeech/model",
        hifigan_model_path="path/to/hifigan/model"
    )

    # Generate audio
    audio_data = tts.generar_audio("Hello, world!")

    # Start WebSocket server
    tts.start_websocket_server()

Dependencies
-----------

- FastSpeech2
- HiFi-GAN
- websockets
- numpy
- soundfile
- pyttsx3

See Also
--------

- :doc:`contextual_activation`
- :doc:`conversation_manager`

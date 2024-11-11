Personality Module
==================

.. automodule:: capibara_model.modules.personality
   :members:
   :undoc-members:
   :show-inheritance:

The Personality Module is a crucial component of the CapibaraGPT model, responsible for managing and applying personality traits to the model's responses. This module enables the model to exhibit consistent personality characteristics across conversations, enhancing the naturalness and coherence of interactions.

Classes
-------

PersonalityModule
^^^^^^^^^^^^^^^^^

.. autoclass:: capibara_model.modules.personality.PersonalityModule
   :members:
   :undoc-members:
   :show-inheritance:

   The main class for managing personality traits and applying them to model responses.

   Key Methods:

   - ``set_personality``: Sets the current personality traits.
   - ``apply_personality``: Applies the current personality to a given response.
   - ``get_current_personality``: Retrieves the current personality settings.

PersonalityTrait
^^^^^^^^^^^^^^^^

.. autoclass:: capibara_model.modules.personality.PersonalityTrait
   :members:
   :undoc-members:
   :show-inheritance:

   Represents individual personality traits that can be applied to the model's behavior.

Usage Example
-------------

.. code-block:: python

   from capibara_model.modules.personality import PersonalityModule, PersonalityTrait

   # Initialize the PersonalityModule
   personality_module = PersonalityModule()

   # Define some personality traits
   friendly = PersonalityTrait("friendly", weight=0.8)
   analytical = PersonalityTrait("analytical", weight=0.6)

   # Set the personality
   personality_module.set_personality([friendly, analytical])

   # Apply personality to a response
   original_response = "The data shows an increasing trend."
   personalized_response = personality_module.apply_personality(original_response)

   print(f"Original response: {original_response}")
   print(f"Personalized response: {personalized_response}")

Personality Application Process
-------------------------------

The PersonalityModule follows these steps to apply personality to responses:

1. **Trait Analysis**: Analyzes the current set of personality traits and their weights.
2. **Response Modification**: Modifies the original response based on the active personality traits.
3. **Language Adjustment**: Adjusts language patterns, vocabulary, and tone to reflect the personality.
4. **Consistency Check**: Ensures that the modified response maintains consistency with previous interactions.

Customizing Personalities
-------------------------

Users can create custom personalities by defining new PersonalityTrait instances and combining them:

.. code-block:: python

   # Create custom personality traits
   humorous = PersonalityTrait("humorous", weight=0.7)
   empathetic = PersonalityTrait("empathetic", weight=0.9)

   # Combine traits for a unique personality
   custom_personality = [humorous, empathetic]

   # Apply the custom personality
   personality_module.set_personality(custom_personality)

Integration with CapibaraGPT
----------------------------

The PersonalityModule is designed to be seamlessly integrated into the CapibaraGPT pipeline:

1. It can be initialized as part of the model's setup process.
2. The `apply_personality` method can be called on generated responses before they are returned to the user.
3. Personality settings can be dynamically adjusted based on user preferences or conversation context.

Notes
-----

- The effectiveness of personality application may vary depending on the complexity of the response and the defined traits.
- It's important to balance personality traits to maintain coherence and avoid exaggerated or unnatural responses.
- Regular evaluation and fine-tuning of personality settings are recommended to ensure optimal performance.

For more detailed information on advanced usage, personality creation guidelines, and integration best practices, please refer to the extended documentation and research papers associated with the CapibaraGPT project.


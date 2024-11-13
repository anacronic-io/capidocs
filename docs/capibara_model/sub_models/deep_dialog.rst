Deep Dialog Model
=================

The `InternalDialogue` model is a Haiku module that simulates an internal dialogue process. It consists of several sub-modules that handle different aspects of the dialogue generation.

Sub-modules
-----------

BasicNeeds
^^^^^^^^^^

The `BasicNeeds` module detects unmet needs in the input using a dense layer and ReLU activation.

CompensatoryNarrative
^^^^^^^^^^^^^^^^^^^^^

The `CompensatoryNarrative` module generates a compensatory narrative for ambiguous inputs by concatenating the input with context and applying a dense layer with tanh activation.

UncertainMemory
^^^^^^^^^^^^^^^

The `UncertainMemory` module stores and organizes inconsistencies or ambiguities in a temporary memory using a dense layer.

BlameNegation
^^^^^^^^^^^^^

The `BlameNegation` module decides between assuming or denying the lack of information based on an uncertainty score calculated using fuzzy logic.

Reconnection
^^^^^^^^^^^^

The `Reconnection` module evaluates the coherence and reconnection with the present input using a dense layer and mean pooling.

InternalDialogue
----------------

The `InternalDialogue` module integrates all the sub-modules to simulate the internal dialogue process. It performs the following steps:

1. Detection of basic needs using the `BasicNeeds` module.
2. Generation of an internal narrative using the `CompensatoryNarrative` module.
3. Storage of inconsistencies in uncertain memory using the `UncertainMemory` module.
4. Adaptive blame and negation using the `BlameNegation` module.
5. Review of coherence and reconnection with the present input using the `Reconnection` module.

The final output is an adaptive combination of the narrative and blame/negation response based on the coherence score.

Initialization and Forward Function
-----------------------------------

The `forward_fn` function defines the forward pass of the `InternalDialogue` model. It takes the input and context as arguments and returns the output and memory.

The model is initialized using Haiku  ``transform`` function, which creates an initialization function and a forward function. The initialization function is called with random keys and example inputs to initialize the model parameters. The forward function is then used to apply the model to new inputs.

Example Usage
-------------

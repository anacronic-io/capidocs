QuineLayer
==========

The `QuineLayer` is the main class that integrates epistemic consistency, analytic-synthetic rejection, and pragmatic adjustment. It processes statements in a given context and provides adjusted responses based on the consistency and relevance of the statements.

Dependencies
------------

The `QuineLayer` and its associated classes depend on the following libraries:

- `sentence_transformers`: Used for loading the embedding model and encoding text.
- `numpy`: Used for numerical operations and computing similarities.
- `functools`: Used for caching embeddings using the `lru_cache` decorator.

Classes
-------

EpistemicInterdependence
^^^^^^^^^^^^^^^^^^^^^^^^

The `EpistemicInterdependence` class evaluates the epistemic consistency of statements using embeddings and cosine similarity. It takes a knowledge base as input and computes the normalized embeddings of the knowledge base.

- `verify_consistency(statement, threshold=0.75)`: Verifies if a statement is consistent with existing knowledge based on the similarity threshold.

AnalyticSyntheticRejection
^^^^^^^^^^^^^^^^^^^^^^^^^^

The `AnalyticSyntheticRejection` class implements the evaluation of contextual relevance of a statement.

- `evaluate_statement(statement, context, threshold=0.6)`: Evaluates the relevance of a statement in a given context based on the similarity threshold.

PragmaticAdjustment
^^^^^^^^^^^^^^^^^^^

The `PragmaticAdjustment` class adjusts responses according to the complexity level.

- `__init__(complexity_level=1)`: Initializes the class with a specified complexity level.
- `adjust_response(response)`: Adjusts the response according to the set complexity level.

QuineLayer
^^^^^^^^^^

The `QuineLayer` class integrates the `EpistemicInterdependence`, `AnalyticSyntheticRejection`, and `PragmaticAdjustment` classes to process statements in a given context.

- `__init__(knowledge_base, complexity_level=1)`: Initializes the class with a knowledge base and complexity level.
- `process_statement(statement, context)`: Processes a single statement in a given context and returns an adjusted response.
- `batch_process_statements(statements, context)`: Processes multiple statements in batch for efficiency and returns a list of adjusted responses.

Utility Functions
-----------------

- `get_embedding(text)`: Retrieves the cached embedding of a text using the loaded embedding model.
- `normalize_text(text)`: Normalizes text by converting it to lowercase and removing extra spaces.

Example Usage
-------------

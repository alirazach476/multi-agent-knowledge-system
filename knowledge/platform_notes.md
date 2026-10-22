# Internal AI Platform Notes

## Retrieval Strategy
We use hybrid retrieval: dense embeddings + keyword filters on metadata tags.
Chunk size of 500-800 tokens with 10-15% overlap works well for policy docs.

## Evaluation
Track faithfulness, answer relevancy, and context precision.
Fail a release if faithfulness drops below 0.85 on the golden set.

## Guardrails
Block answers that are not grounded in retrieved context.
Escalate legal and medical questions to human specialists.

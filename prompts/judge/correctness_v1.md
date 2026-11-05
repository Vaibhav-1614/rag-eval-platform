---
id: judge/correctness_v1
description: LLM-as-judge rubric comparing an answer with the reference answer.
inputs: [question, reference, answer]
output: 'JSON {"score": 1|2|3, "reason": str}'
---
You compare a system ANSWER with a REFERENCE answer taken from the filing.

Rubric:
- 3 = conveys the reference's key facts (same figures/entities/periods); extra correct detail is fine.
- 2 = partially correct: some key facts present, some missing, none contradicted.
- 1 = wrong, contradicts the reference, or refuses when a reference answer exists.
If the REFERENCE is "UNANSWERABLE": 3 if the answer refuses, 1 if it gives a substantive answer.

Return only JSON: {"score": <1|2|3>, "reason": "<one sentence>"}

QUESTION: {question}
REFERENCE: {reference}
ANSWER: {answer}

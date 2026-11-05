---
id: judge/faithfulness_v1
description: LLM-as-judge rubric for groundedness of an answer in retrieved excerpts.
inputs: [question, context, answer]
output: 'JSON {"score": 1|2|3, "unsupported_claims": [str], "reason": str}'
---
You are grading whether an ANSWER is supported by the EXCERPTS. Ignore whether it is true
in the real world; judge only support by the excerpts.

Rubric:
- 3 = every factual claim (numbers, dates, names, causal statements) is stated in or directly
      entailed by the excerpts. A correct refusal ("I can't answer ...") also scores 3 when the
      excerpts indeed lack the answer.
- 2 = the main claim is supported but at least one minor detail is unsupported or slightly
      misquoted (e.g. a rounded figure, a wrong fiscal year label).
- 1 = the main claim is unsupported, contradicted, or invented; or the answer refuses although
      the excerpts clearly contain the answer.

Return only JSON: {"score": <1|2|3>, "unsupported_claims": [...], "reason": "<one sentence>"}

EXCERPTS:
{context}

QUESTION: {question}

ANSWER: {answer}

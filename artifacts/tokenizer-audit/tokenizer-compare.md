# Qwen tokenizer comparison

- A: `/root/autodl-tmp/qwen35-ple/models/Qwen3.5-0.8B`
- B: `/root/autodl-tmp/qwen35-ple/models/Qwen3.8-Flash-Next-FP8-tokenizer`

## Runtime summary

| Field | A | B |
|---|---|---|
| class | Qwen2Tokenizer | Qwen2Tokenizer |
| len | 248077 | 248077 |
| vocab_size | 248044 | 248044 |
| eos_token | <|endoftext|> | <|im_end|> |
| eos_token_id | 248044 | 248046 |
| pad_token | <|endoftext|> | <|endoftext|> |
| pad_token_id | 248044 | 248044 |
| added_vocab_count | 33 | 33 |

## Raw files

| File | A sha256 | B sha256 | Equal |
|---|---|---|---|
| tokenizer.json | fe000e3ed39e | 0997f410c57a | False |
| tokenizer_config.json | e611fbccc7c2 | b11349aafa7c | False |
| vocab.json | ce99b4cb2983 | ce99b4cb2983 | True |
| merges.txt | a9d356d7bdf1 | a9d356d7bdf1 | True |

## Raw tokenizer.json

- model.vocab equal: `True`
- model.merges equal: `True`
- normalizer equal: `True`
- pre_tokenizer equal: `True`
- post_processor equal: `True`
- decoder equal: `True`
- added_tokens A/B: `22` / `33`

## Encoding

| Text | A len | B len | Equal no-special | Equal with special |
|---|---:|---:|---|---|
| `The capital of France is Paris.` | 7 | 7 | True | True |
| `Hello, world! How are you?` | 8 | 8 | True | True |
| `你好，世界！今天天气怎么样？` | 8 | 8 | True | True |
| `def foo(x):\n    return x + 1  # code` | 14 | 14 | True | True |
| `1234567890 3.14159 -42` | 21 | 21 | True | True |
| `café ﬁ ＡＢＣ ｈｅｌｌｏ` | 17 | 17 | True | True |
| `emoji 😀🚀🔥 and symbols ©®™` | 13 | 13 | True | True |
| `multiple   spaces	and\nnewlines` | 7 | 7 | True | True |
| `naïve résumé coöperate` | 7 | 7 | True | True |
| `<think>\nreasoning\n</think> answer` | 7 | 7 | True | True |
| `<tool_response>{"x": 1}</tool_response>` | 8 | 8 | True | True |
| `<\|im_start\|>user\nhi<\|im_end\|>` | 5 | 5 | True | True |
| `<\|endoftext\|>` | 1 | 1 | True | True |
| `<\|audio_start\|>abc<\|audio_end\|>` | 3 | 3 | True | True |

## Verdict

- base_vocab_identical: `True`
- merges_identical: `True`
- normalizer_identical: `True`
- pre_tokenizer_identical: `True`
- post_processor_identical: `True`
- decoder_identical: `True`
- runtime_vocab_identical: `True`
- runtime_eos_ids_equal: `False`
- ordinary_text_encoding_identical: `True`
- all_text_encoding_identical_with_special_tokens: `True`

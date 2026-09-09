# Round 140：Qwen3.5 vs Qwen3.8-Flash-Next tokenizer 对齐审计

> 日期：2026-09-09  
> 目的：确认 Qwen3.8-Flash-Next PLE 的 raw-token rowid 映射与 Qwen3.5-0.8B tokenizer 兼容  
> 产物：`artifacts/tokenizer-audit/tokenizer-compare.{json,md}`

---

## 0. 背景

Qwen3.8-Flash-Next PLE 使用 **raw token id** 做 n-gram hashing：

```text
Qwen token id
  -> PLE_QWEN_V1 n-gram hash
  -> rowid
  -> qwen38-rows
```

它不像 DeepSeek Engram 那样先做 CompressedTokenizer 文本规范化/词表压缩。  
因此 tokenizer 是否对齐会直接影响 PLE 行读取是否正确。

本轮审计比较：

```text
A: /root/autodl-tmp/qwen35-ple/models/Qwen3.5-0.8B
B: /root/autodl-tmp/qwen35-ple/models/Qwen3.8-Flash-Next-FP8-tokenizer
```

B 的 tokenizer 文件从 ModelScope 的 `Qwen/Qwen3.8-Flash-Next-FP8` 下载：

```text
tokenizer.json
tokenizer_config.json
vocab.json
merges.txt
config.json
```

---

## 1. 文件级比较

| 文件 | Qwen3.5 sha256 | Qwen3.8 sha256 | 相同 |
|---|---|---|---|
| `vocab.json` | `ce99b4cb2983...` | `ce99b4cb2983...` | ✅ |
| `merges.txt` | `a9d356d7bdf1...` | `a9d356d7bdf1...` | ✅ |
| `tokenizer.json` | `fe000e3ed39e...` | `0997f410c57a...` | ❌ |
| `tokenizer_config.json` | `e611fbccc7c2...` | `b11349aafa7c...` | ❌ |

`tokenizer.json` / `tokenizer_config.json` 不同，但核心 BPE 数据相同：

```text
model.vocab  equal: True
model.merges equal: True
normalizer   equal: True
pre_tokenizer equal: True
post_processor equal: True
decoder      equal: True
```

`tokenizer.json` 唯一的结构差异：

```text
model.ignore_merges: Qwen3.5 缺省(None) / Qwen3.8 False
```

两者实际语义一致：都不忽略 merges。

---

## 2. Runtime tokenizer 比较

用 `AutoTokenizer.from_pretrained(..., local_files_only=True)` 加载后：

| 字段 | Qwen3.5 | Qwen3.8 |
|---|---:|---:|
| class | `Qwen2Tokenizer` | `Qwen2Tokenizer` |
| `len(tokenizer)` | 248077 | 248077 |
| `vocab_size` | 248044 | 248044 |
| `eos_token` | `<|endoftext|>` | `<|im_end|>` |
| `eos_token_id` | **248044** | **248046** |
| `pad_token` | `<|endoftext|>` | `<|endoftext|>` |
| `pad_token_id` | 248044 | 248044 |
| `added_vocab_count` | 33 | 33 |
| runtime vocab equal | — | ✅ True |

### 2.1 added tokens

`tokenizer.json` 原始 `added_tokens`：

```text
Qwen3.5: 22
Qwen3.8: 33
```

Qwen3.8 多出的 11 个：

```text
248066 <tool_response>
248067 </tool_response>
248068 <think>
248069 </think>
248070 <|audio_start|>
248071 <|audio_end|>
248072 <tts_pad>
248073 <tts_text_bos>
248074 <tts_text_eod>
248075 <tts_text_bos_single>
248076 <|audio_pad|>
```

关键：Qwen3.5 的 `tokenizer_config.json` 里也有这 11 个
`added_tokens_decoder`，所以 **AutoTokenizer 运行时两边都是 33 个 added tokens**，
runtime vocab 完全一致。

如果只加载 `tokenizer.json`（例如 `tokenizers.Tokenizer.from_file`），
才会看到 22 vs 33 的差异；我们当前训练/评测使用的是 `AutoTokenizer`，
不触发这个问题。

### 2.2 special tokens

两边 shared special tokens 一致：

```text
<|endoftext|>, <|audio_start|>, <|audio_end|>, <|audio_pad|>,
<|image_pad|>, <|video_pad|>, <|vision_start|>, <|vision_end|>
```

唯一差异：

```text
Qwen3.5 eos = <|endoftext|> = 248044
Qwen3.8 eos = <|im_end|>    = 248046
```

`<|im_end|>` 在两边都是 special token，只是 Qwen3.5 不把它设为 `eos_token`。

### 2.3 tokenizer_config 差异

除 `eos_token` 外，只有：

```text
chat_template:
  Qwen3.5 7755 chars
  Qwen3.8 8952 chars
```

chat template 不参与普通文本 tokenization，也不影响 PLE rowid。

---

## 3. 同一段文本的 token id 比较

测试覆盖：

- 英文、中文；
- 代码、数字；
- 多空格、tab、换行；
- Unicode：`café`、`ﬁ`、全角 `ＡＢＣ`、emoji；
- `<think>`、`</think>`、`<tool_response>`；
- `<|im_start|>`、`<|im_end|>`、`<|endoftext|>`；
- `<|audio_start|>`、`<|audio_end|>`。

结果：

```text
all no-special encodings identical : True
all with-special encodings identical: True
```

即：

```text
AutoTokenizer(Qwen3.5) 与 AutoTokenizer(Qwen3.8)
在测试文本上产生完全相同的 input_ids
```

因为 PLE rowid 只依赖 token id，所以：

```text
rowids_for_seq(ids_qwen35) == rowids_for_seq(ids_qwen38)
```

对普通语料和上述 special-token 文本都成立。

---

## 4. 对 PLE 的含义

### 4.1 结论

```text
Qwen3.5 tokenizer 与 Qwen3.8-Flash-Next tokenizer
在 PLE 所需的 raw token id 语义上兼容。
```

具体来说：

- base vocab 248044 完全一致；
- merges 完全一致；
- normalizer / pre_tokenizer / post_processor / decoder 完全一致；
- runtime added vocab 完全一致（33 个）；
- 代表性文本 token id 完全一致。

因此当前使用 Qwen3.5 tokenizer 生成 token ids、再走 `PLE_QWEN_V1` raw-token rowid 映射的做法是合理的，不需要因为 tokenizer 差异重新抽表。

### 4.2 EOS 差异不是 PLE 错误

Qwen3.8 的 tokenizer `eos_token` 是 `<|im_end|>` = 248046，  
而 Qwen PLE 的 context reset 常量是：

```text
PLE_QWEN_EOS = 248044 = <|endoftext|>
```

这是两个不同层面的 EOS：

```text
生成停止 EOS（tokenizer/model 配置）
  !=
PLE 分段 EOS（PLE_QWEN_V1 规范常量）
```

Qwen PLE 规范明确使用 248044 做 n-gram 上下文分段，  
所以我们的 `ple_hash.py` / EngramDB rowid 映射保持 248044 是正确的。

如果未来改用 Qwen3.8 tokenizer 做生成，需要单独处理：

- 生成停止：使用 tokenizer 的 eos 248046；
- PLE 分段：仍然使用 248044；
- 两者不能混为一个变量。

---

## 5. 审计脚本

新增：

```text
scripts/compare_qwen_tokenizers.py
```

运行方式：

```bash
PYTHONPATH=src python scripts/compare_qwen_tokenizers.py \
  --tokenizer-a /root/autodl-tmp/qwen35-ple/models/Qwen3.5-0.8B \
  --tokenizer-b /root/autodl-tmp/qwen35-ple/models/Qwen3.8-Flash-Next-FP8-tokenizer \
  --output outputs/tokenizer-compare.json \
  --markdown outputs/tokenizer-compare.md
```

脚本会输出：

- 文件 sha256；
- raw `tokenizer.json` 结构差异；
- runtime vocab / special token / added vocab；
- 固定测试文本的 token id 比较；
- verdict JSON。

---

## 6. 结论

> **Qwen3.5-0.8B 与 Qwen3.8-Flash-Next 的 tokenizer 在 PLE raw-token 语义上对齐。**
>
> 没有 merge 差异，没有 runtime added-token 差异，普通文本与测试 special-token 文本的 token id 完全一致。
>
> 唯一需要记住的是：Qwen3.8 tokenizer 的生成 EOS 是 248046，而 PLE 分段 EOS 固定是 248044；这两个概念不能混淆。

import re
from collections import Counter

def BPE_Tokenizer_Training(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
  # 初始化词汇表
  vocab = {i: bytes([i]) for i in range(256)}
  # 添加特殊标记
  for token in special_tokens:
    token_bytes = token.encode("utf-8")
    if token_bytes not in vocab.values():
      vocab[len(vocab)] = token_bytes
  PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
  with open(input_path,'r', encoding="utf-8") as f:
    text = f.read()
  matches = re.finditer(PAT, text, re.UNICODE)
  # 提取匹配的单词
  tokens = [match.group() for match in matches]
  # 统计词频
  word_counts = dict(Counter(tokens))
  # 将单词转换为字节数组
  byte_word_counts = {word.encode("utf-8"): count for word, count in word_counts.items()}
  # 合并记录
  merges = []

  # 选择并合并字节对
  while len(vocab) < vocab_size:
    # 统计字节对频率
    byte_pair_freq = get_pair_statistics(byte_word_counts)
    # 选择频率最高的字节对
    best_pair = max(byte_pair_freq, key=byte_pair_freq.get)
    # 合并字节对
    new_token = best_pair[0] + best_pair[1]
    # 更新词汇表
    vocab[len(vocab)] = new_token
    # 更新合并记录
    merges.append(best_pair)

    # 更新 byte_word_counts 以反映合并后的词汇表
    new_byte_word_counts = Counter()
    for byte_word, count in byte_word_counts.items():
        # 使用 bytes.replace 方法替换字节对
        new_byte_word = byte_word.replace(best_pair[0] + best_pair[1], new_token)
        new_byte_word_counts[new_byte_word] += count
    byte_word_counts = new_byte_word_counts

  return vocab,merges



def get_pair_statistics(
    byte_word_counts: dict[bytes, int]
) -> dict[tuple[bytes, bytes], int]:
  byte_pair_freq = Counter()

  for byte_word, count in byte_word_counts.items():
    # 提取字节对
    for i in range(len(byte_word) - 1):
        byte_pair = (byte_word[i:i+1], byte_word[i+1:i+2])
        byte_pair_freq[byte_pair] += count
  
  return dict(byte_pair_freq)
import regex as re
from collections import Counter
import collections

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
  word_counts = { ' '.join(word): count for word, count in word_counts.items() }
  # 合并记录
  merges = []
  # 选择并合并字节对
  while len(vocab) < vocab_size:
    # 统计字节对频率
    pair_freq = get_pair_statistics(word_counts)
    # 如果没有可合并的字节对，退出循环
    if not pair_freq:
      print("No more pairs to merge. Exiting the loop.")
      break
    print(pair_freq)
    # 选择频率最高的字节对
    best_pair = max(pair_freq, key=pair_freq.get)
    print(best_pair)
    # 合并字节对
    new_token = best_pair[0] + best_pair[1]
    # 更新词汇表
    vocab[len(vocab)] = new_token
    # 更新合并记录
    merges.append(best_pair)

    # 更新 byte_word_counts 以反映合并后的词汇表
    new_word_counts = Counter()
    for word, count in word_counts.items():
        # 使用 bytes.replace 方法替换字节对
        new_word = word.replace(best_pair[0] + " " + best_pair[1], new_token)
        new_word_counts[new_word] += count
        print(f"Replacing {best_pair[0] + best_pair[1]} with {new_token} in {word} -> {new_word}")
    word_counts = new_word_counts
    print(word_counts)

  return vocab,merges



def get_pair_statistics(
    word_counts: dict[str, int],
) -> dict[tuple[bytes, bytes], int]:
  pairs_freq = collections.defaultdict(int)
  for word, freq in word_counts.items():
      symbols = word.split()
      for i in range(len(symbols)-1):
          pairs_freq[symbols[i],symbols[i+1]] += freq
  return pairs_freq


if __name__ == "__main__":
      # 测试数据
    input_path = "../data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    # 调用函数
    vocab, merges = BPE_Tokenizer_Training(input_path, vocab_size, special_tokens)

    # 输出结果
    print("Vocabulary:")
    for k, v in vocab.items():
        print(f"{k}: {v}")

    print("\nMerges:")
    for merge in merges:
        print(merge)
    #  # 测试数据
    # input_path = "test_input.txt"
    # vocab_size = 300
    # special_tokens = ["<s>", "</s>"]

    # # 调用函数
    # vocab, merges = BPE_Tokenizer_Training(input_path, vocab_size, special_tokens)

    # # 输出结果
    # print("Vocabulary:" + str(len(vocab)))
    # for k, v in vocab.items():
    #     print(f"{k}: {v}")

    # print("\nMerges:")
    # for merge in merges:
    #     print(merge)
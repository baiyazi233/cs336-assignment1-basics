import regex as re
from collections import Counter

# 初始化词汇表
def initialize_vocab(special_tokens):
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in vocab.values():
            vocab[len(vocab)] = token_bytes
    return vocab

# 统计字节对频率
def get_pair_statistics(byte_word_counts):
    byte_pair_freq = Counter()
    for byte_word, count in byte_word_counts.items():
        for i in range(len(byte_word) - 1):
            byte_pair = (byte_word[i:i+1], byte_word[i+1:i+2])
            byte_pair_freq[byte_pair] += count
    return dict(byte_pair_freq)

# 合并字节对
def merge_pair(best_pair, byte_word_counts, vocab):
    new_token = best_pair[0] + best_pair[1]
    new_byte_word_counts = Counter()
    for byte_word, count in byte_word_counts.items():
        new_byte_word = byte_word.replace(best_pair[0] + best_pair[1], new_token)
        new_byte_word_counts[new_byte_word] += count
    vocab[len(vocab)] = new_token
    return new_byte_word_counts, vocab

# BPE 分词器训练
def BPE_Tokenizer_Training(input_path, vocab_size, special_tokens, max_merges=None):
    vocab = initialize_vocab(special_tokens)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(input_path, 'r', encoding="utf-8") as f:
        text = f.read()
    matches = re.finditer(PAT, text, re.UNICODE)
    tokens = [match.group() for match in matches]
    word_counts = dict(Counter(tokens))
    byte_word_counts = {word.encode("utf-8"): count for word, count in word_counts.items()}
    merges = []

    while len(vocab) < vocab_size and (max_merges is None or len(merges) < max_merges):
        byte_pair_freq = get_pair_statistics(byte_word_counts)
        if not byte_pair_freq:
            break
        best_pair = max(byte_pair_freq, key=byte_pair_freq.get)
        byte_word_counts, vocab = merge_pair(best_pair, byte_word_counts, vocab)
        merges.append(best_pair)

    return vocab, merges

# 测试代码
if __name__ == "__main__":
    input_path = "test_input.txt"
    vocab_size = 300
    special_tokens = ["<s>", "</s>"]
    max_merges = 6  # 限制合并次数

    # 创建测试输入文件
    with open(input_path, "w", encoding="utf-8") as f:
        f.write("This is a test sentence for BPE tokenizer training.\n")
        f.write("BPE is a popular method for subword tokenization.\n")
        f.write("It helps in handling out-of-vocabulary words.\n")

    # 调用函数
    vocab, merges = BPE_Tokenizer_Training(input_path, vocab_size, special_tokens, max_merges)

    # 输出结果
    print("Vocabulary:")
    for k, v in vocab.items():
        print(f"{k}: {v}")

    print("\nMerges:")
    for merge in merges:
        print(f"{merge[0].decode('utf-8')} {merge[1].decode('utf-8')}")
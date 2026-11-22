---
id: bpe
module: transformers
title: Tokenization and byte-pair encoding
minutes: 40
summary: A vocabulary built by repeatedly gluing the pair you see most often.
idea: A tokenizer is a compression of characters that tries to keep frequent chunks whole.
---

## Intuition

A language model does not see letters unless you decide that it does. It sees token ids. The vocabulary is a design choice with a cost: a small vocabulary makes sequences long, and a large one makes the softmax wide. Byte-pair encoding sits in between. Start from characters. Count every adjacent pair in the corpus. Glue the most frequent pair into a new symbol. Repeat. Frequent words collapse into one or two tokens. Rare words fall back to smaller pieces, including characters, so you never face a truly unknown string.

The end-of-word marker matters. Without it, a pair that straddles a word boundary looks the same as a pair inside a word. This lesson appends the symbol `</w>` to every word before counting. A merge is allowed to absorb that marker, which is how a whole word becomes one token.

Ties happen in tiny corpora. Break them deterministically: among pairs with the highest count, choose the lexicographically smallest pair, comparing the left symbol and then the right. The new symbol is the concatenation of the two pieces. `a` merged with `b` is the symbol `ab`, not a fresh integer. That choice keeps the exercise readable. Real tokenizers use integer ids and byte-level alphabets. The loop is the same.

Encoding a new word applies the learned merges in the order they were learned, not by recounting frequencies. The training counts do not travel with the tokenizer. The merge list does.

## Math

A corpus is a multiset of words. Represent word $w$ as the tuple of its characters followed by `</w>`. The pair count is

$$
\mathrm{count}(x, y) = \sum_{\text{words}} \sum_{i} \mathbf{1}[t_i = x \land t_{i+1} = y].
$$

One merge replaces every non-overlapping occurrence of the chosen pair, scanning each word from the left. If the same pair occurs three times in a row, two symbols merge and the third waits, because the scan jumps by two. After `num_merges` successful merges, or earlier if no pair remains, the algorithm stops.

The training result in this lesson is the list of triples `(left, right, merged)` in the order the merges were chosen.

## Resources

- [Neural machine translation of rare words with subword units](https://aclanthology.org/P16-1162/) · paper · Sennrich, Haddow, and Birch · The paper that put byte-pair encoding into neural translation.
- [Byte-pair encoding tokenization](https://huggingface.co/learn/llm-course/chapter6/5) · course · Hugging Face LLM Course · A visual walk through the merge loop and the encode step.
- [minbpe](https://github.com/karpathy/minbpe) · code · Andrej Karpathy · A minimal BPE implementation, including the byte-level version used by GPT-style models.
- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) · video · Andrej Karpathy · The tokenizer lecture on the Zero to Hero page. Watch it beside minbpe.
- [LLMs from scratch, chapter 2](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02) · code · Sebastian Raschka · Working with text data, from raw strings to token ids and batches.

## Exercise

Implement `train_bpe(words, num_merges)` and `encode(word, merges)`.

`words` is a list of strings. Repeated entries are repeated words. Each word becomes characters plus the end marker `</w>`. Count pairs across the whole list. On a tie, pick the lexicographically smallest pair. The merged symbol is `left + right`. Stop early if no pairs remain. Return a list of `(left, right, merged)` tuples.

`encode(word, merges)` starts from the characters of a single word plus `</w>` and applies every merge in order, left to right, repeatedly within the word until that pair is gone. Return a tuple of symbols.

On `words = ["ab", "ab", "abc", "cab"]` and `num_merges = 3`, the merges are `("a","b","ab")`, then `("ab","</w>","ab</w>")`, then `("ab","c","abc")`. `encode("ab", merges)` is `("ab</w>",)`. `encode("abc", merges)` is `("abc", "</w>")`.

```python starter
def train_bpe(words, num_merges):
    raise NotImplementedError


def encode(word, merges):
    raise NotImplementedError
```

```python solution
def _pairs(tokens):
    counts = {}
    for word in tokens:
        for left, right in zip(word, word[1:]):
            counts[(left, right)] = counts.get((left, right), 0) + 1
    return counts


def _merge(tokens, pair, new_symbol):
    merged_words = []
    for word in tokens:
        merged = []
        index = 0
        while index < len(word):
            if index < len(word) - 1 and word[index] == pair[0] and word[index + 1] == pair[1]:
                merged.append(new_symbol)
                index += 2
            else:
                merged.append(word[index])
                index += 1
        merged_words.append(tuple(merged))
    return merged_words


def train_bpe(words, num_merges):
    tokens = [tuple(list(word) + ["</w>"]) for word in words]
    merges = []
    for _ in range(int(num_merges)):
        counts = _pairs(tokens)
        if not counts:
            break
        pair = min(counts, key=lambda item: (-counts[item], item[0], item[1]))
        symbol = pair[0] + pair[1]
        tokens = _merge(tokens, pair, symbol)
        merges.append((pair[0], pair[1], symbol))
    return merges


def encode(word, merges):
    tokens = tuple(list(word) + ["</w>"])
    for left, right, symbol in merges:
        tokens = _merge([tokens], (left, right), symbol)[0]
    return tokens
```

```python tests
words = ["ab", "ab", "abc", "cab"]
merges = train_bpe(words, 3)
assert merges == [("a", "b", "ab"), ("ab", "</w>", "ab</w>"), ("ab", "c", "abc")]
assert encode("ab", merges) == ("ab</w>",)
assert encode("abc", merges) == ("abc", "</w>")
assert encode("zz", merges) == ("z", "z", "</w>")
assert train_bpe([""], 4) == []
```

## Quiz

### q1
BPE chooses the next merge by

- [ ] the longest remaining string in the vocabulary.
- [x] the adjacent pair with the highest count, with a fixed tie-break.
- [ ] the pair that maximizes validation likelihood.
- [ ] a random pair, so that the vocabulary covers rare words.

> The algorithm is frequentist and greedy. It does not look at a language-model loss. The tie-break only exists so two correct programs agree.

### q2
The end-of-word marker is there so that

- [ ] every word has an odd number of symbols.
- [x] a pair inside a word is not confused with a pair that crosses a boundary, and a whole word can become one symbol.
- [ ] decoding can ignore spaces.
- [ ] the softmax has a reserved index.

> Without a boundary symbol, the statistics of "the" and of a "th" that happens to sit at the end of one word and the start of the next get mixed.

### q3
Encoding applies merges

- [ ] by recounting pairs in the new word and merging the current mode.
- [x] in the recorded training order, whether or not that pair is frequent in the new word.
- [ ] only to words that appeared in the training list.
- [ ] from the end of the word backward.

> The merge list is the tokenizer. A new word does not get to vote on a new vocabulary.

---
id: edit-distance
module: classic
title: Dynamic programming and edit distance
minutes: 35
summary: The cheapest way to turn one string into another, built from smaller suffixes.
idea: Sequence problems often have a table: each cell reuses three neighbors.
---

## Intuition

Edit distance is the smallest number of single-character edits that turn one string into another. An edit is an insertion, a deletion, or a substitution. "kitten" becomes "sitting" in three edits, and you cannot do it in two. The number shows up in spelling correction, in DNA alignment, and as a sanity check on generated text. It is in this course because the method is the one you will see again whenever a sequence is scored against another sequence.

Trying every edit sequence is exponential. The useful observation is smaller: the best way to align the first $i$ characters of $a$ with the first $j$ characters of $b$ depends only on three shorter problems. You can delete $a_i$, insert $b_j$, or substitute (or copy, if the characters match). Store those answers in a table and each cell is a constant-time lookup. That is dynamic programming. The table is not an optimization trick bolted on afterwards. It is the definition of the optimum, written so that a computer can fill it.

Read the table, not just the final number, when a distance surprises you. The last cell is the cost. A path of choices through the table is an alignment. This exercise only asks for the cost. Once that is right, reconstructing the path is a walk backward from the corner.

Tokenizers and attention will not use Levenshtein distance directly. They will use the same habit: define a quantity on prefixes, find the recurrence, fill a table or a tensor in the order the recurrence demands.

## Math

Let $D_{i,j}$ be the edit distance between the first $i$ characters of $a$ and the first $j$ characters of $b$. The base cases are the empty prefixes:

$$
D_{0,0} = 0, \qquad D_{i,0} = i, \qquad D_{0,j} = j.
$$

The recurrence, with unit costs, is

$$
D_{i,j} = \min\begin{cases}
D_{i-1,j} + 1 \\
D_{i,j-1} + 1 \\
D_{i-1,j-1} + [a_i \neq b_j]
\end{cases}
$$

where $[a_i \neq b_j]$ is 0 when the characters match and 1 otherwise. The answer is $D_{|a|,|b|}$. Time and memory are $\Theta(|a|\,|b|)$ for the full table. This lesson keeps the full table so the recurrence stays visible.

## Resources

- [Neural machine translation of rare words with subword units](https://aclanthology.org/P16-1162/) · paper · Sennrich, Haddow, and Birch · Byte-pair encoding, the tokenizer lesson, grew up next to this kind of string problem.
- [CS231n Python NumPy tutorial](https://cs231n.github.io/python-numpy-tutorial/) · reading · Stanford CS231n · Enough Python to write the table without fighting the language.
- [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch) · code · Sebastian Raschka · Chapter 2 works with text as sequences. The edit-distance habit is the warm-up.
- [Stanford CS224N](https://web.stanford.edu/class/cs224n/) · course · Stanford · The NLP course that treats sequence structure as the main object, not a preprocessing step.

## Exercise

Implement `edit_distance(a, b) -> int`.

Use unit-cost insertion, deletion, and substitution. Matching characters cost 0. Return the distance, not the alignment. `edit_distance("", "")` is 0. `edit_distance("", "abc")` is 3. The classic pair `edit_distance("kitten", "sitting")` is 3.

A Python list of lists is enough. You do not need NumPy.

```python starter
def edit_distance(a, b):
    raise NotImplementedError
```

```python solution
def edit_distance(a, b):
    a = str(a)
    b = str(b)
    n, m = len(a), len(b)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        table[i][0] = i
    for j in range(m + 1):
        table[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            substitute = 0 if a[i - 1] == b[j - 1] else 1
            table[i][j] = min(
                table[i - 1][j] + 1,
                table[i][j - 1] + 1,
                table[i - 1][j - 1] + substitute,
            )
    return int(table[n][m])
```

```python tests
assert edit_distance("", "") == 0
assert edit_distance("", "abc") == 3
assert edit_distance("abc", "") == 3
assert edit_distance("abc", "abc") == 0
assert edit_distance("kitten", "sitting") == 3
assert edit_distance("a", "b") == 1
assert edit_distance("flaw", "lawn") == 2
assert edit_distance("intention", "execution") == 5
```

## Quiz

### q1
$D_{i,0}$ equals $i$ because

- [ ] the first character of $b$ is always a match.
- [x] turning $i$ characters into the empty string takes $i$ deletions.
- [ ] dynamic programming requires a zero column for numerical stability.
- [ ] substitution of an empty character costs $i$.

> The empty prefix of $b$ can only be reached by deleting every character of the prefix of $a$. Each deletion costs 1.

### q2
In the recurrence, a matching pair of characters contributes

- [ ] a forced cost of 1, because you still consume both characters.
- [x] a cost of 0 added to the diagonal neighbor $D_{i-1,j-1}$.
- [ ] nothing. Matches are handled only in the base case.
- [ ] a cost equal to the length of the longer string.

> You may align the two equal characters and pay nothing. The other two options, insert and delete, still cost 1, so the min will pick the diagonal when that is best.

### q3
Why is the table polynomial instead of exponential?

- [ ] Strings are short, so the constant factors hide the exponential.
- [ ] Edit distance is always the length of the longer string.
- [x] Each prefix pair is solved once and reused, instead of being recomputed inside every edit sequence.
- [ ] Python lists are faster than recursion by an exponential factor.

> There are $(|a|+1)(|b|+1)$ distinct subproblems. The exponential blow-up was the repeated solution of the same prefixes, not the existence of three choices.

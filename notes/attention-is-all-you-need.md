# Attention Is All You Need

**arXiv ID:** [1706.03762](https://arxiv.org/abs/1706.03762)

**Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (Google Brain / Google Research / University of Toronto) — NIPS 2017.

Reading notes for lab journal club. Every number below is taken directly from the paper, and the walkthrough follows the paper's own section structure: Model Architecture (§3) → Why Self-Attention (§4) → Training (§5) → Results (§6).

---

## Summary (from the abstract, in my own words)

Before this paper, the best sequence transduction models (e.g., for machine translation) were encoder–decoder networks built on recurrent or convolutional layers, and the strongest of them already used an attention mechanism to connect the encoder to the decoder. The Transformer's proposal is radical simplicity: drop recurrence and convolutions **entirely** and build the whole model out of attention mechanisms alone. On two WMT 2014 translation benchmarks the result is models that are better in quality while being far more parallelizable and much cheaper to train. The headline numbers: **28.4 BLEU on WMT 2014 English→German**, improving over the existing best results — including ensembles — by more than 2 BLEU, and a **new single-model state-of-the-art 41.8 BLEU on WMT 2014 English→French**, reached after training for just 3.5 days on eight GPUs, a small fraction of the training cost of the best models in the literature. The authors also argue the architecture generalizes beyond translation by applying it successfully to English constituency parsing, both with large and with limited training data.

---

## Model architecture (paper §3)

### Encoder and decoder stacks (§3.1)

The model keeps the familiar encoder–decoder shape: the encoder maps an input sequence of symbol representations $(x_1, \dots, x_n)$ to a sequence of continuous representations $\mathbf{z} = (z_1, \dots, z_n)$; the decoder then generates the output sequence $(y_1, \dots, y_m)$ one element at a time, auto-regressively consuming the previously generated symbols as additional input.

- **Encoder:** a stack of $N = 6$ identical layers. Each layer has **two sub-layers**: (1) a multi-head **self-attention** mechanism, and (2) a simple, position-wise fully connected **feed-forward** network. A residual connection wraps each sub-layer, followed by layer normalization, i.e. the output of each sub-layer is $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$. To make the residual connections possible, all sub-layers — and the embedding layers — produce outputs of dimension $d_{\text{model}} = 512$.
- **Decoder:** also a stack of $N = 6$ identical layers. In addition to the encoder's two sub-layers, each decoder layer **inserts a third sub-layer** that performs multi-head attention over the output of the encoder stack (encoder–decoder attention). Residual connections plus layer normalization are used the same way. The decoder's self-attention sub-layer is modified so that positions cannot attend to subsequent positions; this masking, combined with the output embeddings being offset by one position, ensures the prediction for position $i$ depends only on the known outputs at positions $< i$.

### How attention works in this model (§3.2)

An attention function maps a **query** and a set of **key–value pairs** to an output: the output is a weighted sum of the values, where the weight on each value comes from a compatibility function of the query with the corresponding key.

**Scaled dot-product attention (§3.2.1).** Queries and keys have dimension $d_k$, values have dimension $d_v$. You take dot products of the query with all keys, **divide each by $\sqrt{d_k}$**, and apply a softmax to obtain the weights on the values:

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

The $1/\sqrt{d_k}$ scaling is what distinguishes it from ordinary dot-product attention: for large $d_k$ the dot products grow large in magnitude (with mean-0, variance-1 components, $q \cdot k$ has variance $d_k$), pushing the softmax into regions with extremely small gradients. Scaling counteracts this. Dot-product attention is otherwise preferred over additive attention because it runs on highly optimized matrix multiplication code.

**Multi-head attention (§3.2.2).** Instead of one attention function with $d_{\text{model}}$-dimensional queries, keys and values, the model linearly projects queries, keys and values $h$ times with different learned projections (to $d_k$, $d_k$, $d_v$ dimensions respectively), runs scaled dot-product attention in parallel on each, concatenates the $h$ results, and projects them once more:

$$\mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)\,W^O$$

This lets the model jointly attend to information from different representation subspaces at different positions — something averaging in a single head inhibits. The model uses $h = 8$ parallel heads with $d_k = d_v = d_{\text{model}}/h = 64$; because each head is low-dimensional, the total cost is similar to single-head attention with full dimensionality.

**Where attention is used (§3.2.3).** Three ways:

1. **Encoder–decoder attention layers** — queries come from the previous decoder layer; keys and values come from the output of the encoder, so every decoder position can attend over all input positions.
2. **Encoder self-attention** — keys, values and queries all come from the previous encoder layer; each position attends to all positions in the previous layer of the encoder.
3. **Decoder self-attention** — each position attends to all decoder positions up to and including that position. Illegal (future) connections are masked by setting their values to $-\infty$ before the softmax, preserving the auto-regressive property.

### Position-wise feed-forward networks (§3.3)

Each encoder and decoder layer also contains a fully connected feed-forward network applied to each position separately and identically — two linear transformations with a ReLU in between:

$$\mathrm{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

The same transformations are used across positions within a layer but different parameters are used from layer to layer (equivalently, two convolutions with kernel size 1). Input and output dimensionality is $d_{\text{model}} = 512$; the **inner-layer dimensionality is $d_{ff} = 2048$**.

### Embeddings and softmax (§3.4)

Learned embeddings convert input and output tokens to vectors of dimension $d_{\text{model}}$; a learned linear transformation plus softmax converts decoder output to next-token probabilities. The same weight matrix is shared between the two embedding layers and the pre-softmax linear transformation, and in the embedding layers those weights are multiplied by $\sqrt{d_{\text{model}}}$.

### Positional encoding (§3.5)

Because there is no recurrence and no convolution, positional information must be injected explicitly. Fixed sinusoidal encodings are added to the input embeddings at the bottom of both stacks:

$$PE_{(pos, 2i)} = \sin\!\left(pos / 10000^{2i/d_{\text{model}}}\right), \qquad PE_{(pos, 2i+1)} = \cos\!\left(pos / 10000^{2i/d_{\text{model}}}\right)$$

Each dimension corresponds to a sinusoid, with wavelengths forming a geometric progression from $2\pi$ to $10000 \cdot 2\pi$. The sinusoidal form was chosen because $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$, making it easy to learn to attend by relative position, and because it may extrapolate to sequence lengths longer than those seen in training. Learned positional embeddings performed nearly identically (Table 2, row (E)).

### Base model — key dimensions

| Quantity | Value |
|---|---|
| Layers per stack, $N$ | 6 (encoder) and 6 (decoder) |
| Model dimension, $d_{\text{model}}$ | 512 |
| Attention heads, $h$ | 8 (with $d_k = d_v = 64$) |
| Feed-forward inner dimension, $d_{ff}$ | 2048 |

---

## Why self-attention (paper §4)

The paper compares self-attention layers to recurrent and convolutional layers along three desiderata: **total computational complexity per layer**, **amount of computation that can be parallelized** (minimum number of sequential operations), and **maximum path length between long-range dependencies** — since shorter paths between any combination of input and output positions make long-range dependencies easier to learn.

| Layer type | Complexity per layer | Sequential operations | Maximum path length |
|---|---|---|---|
| Self-attention | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k(n))$ |
| Self-attention (restricted) | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n/r)$ |

($n$ = sequence length, $d$ = representation dimension, $k$ = kernel size of convolutions, $r$ = neighborhood size in restricted self-attention.)

Key points from the comparison:

- A self-attention layer connects all positions with a **constant number of sequentially executed operations**, whereas a recurrent layer requires $O(n)$ sequential operations.
- Self-attention is faster than recurrence when $n < d$, which is the usual case for state-of-the-art word-piece and byte-pair sentence representations.
- A single convolutional layer with kernel width $k < n$ does **not** connect all pairs of input and output positions; doing so requires $O(n/k)$ contiguous layers or $O(\log_k(n))$ dilated convolution layers, lengthening the longest paths. Even separable convolutions (complexity $O(k \cdot n \cdot d + n \cdot d^2)$) at $k = n$ equal the combination of a self-attention layer and a point-wise feed-forward layer — exactly the approach this model takes.
- As a side benefit, self-attention yields more interpretable models: individual attention heads learn different tasks, and many exhibit behavior related to syntactic and semantic structure (visualized in the appendix).

---

## Training setup (paper §5)

### Data and batching (§5.1)

- **English→German:** standard WMT 2014 English–German dataset, about 4.5 million sentence pairs, byte-pair encoding with a shared source–target vocabulary of about 37,000 tokens.
- **English→French:** the significantly larger WMT 2014 English–French dataset of 36M sentences, with a 32,000 word-piece vocabulary.
- Sentence pairs were batched by approximate sequence length, each batch containing roughly 25,000 source tokens and 25,000 target tokens.

### Hardware and schedule (§5.2)

Models were trained on **one machine with 8 NVIDIA P100 GPUs**:

- **Base models:** each training step took about **0.4 seconds**; trained for a total of **100,000 steps, or 12 hours**.
- **Big models:** step time **1.0 seconds**; trained for **300,000 steps (3.5 days)**.

### Optimizer and warmup schedule (§5.3)

**Adam** with $\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$. The learning rate varies over the course of training:

$$lrate = d_{\text{model}}^{-0.5} \cdot \min\left(step\_num^{-0.5},\; step\_num \cdot warmup\_steps^{-1.5}\right)$$

This corresponds to **increasing the learning rate linearly for the first $warmup\_steps$ training steps, and decreasing it thereafter proportionally to the inverse square root of the step number**. The paper uses $warmup\_steps = 4000$.

### Regularization (§5.4)

Three types:

1. **Residual dropout** — applied to the output of each sub-layer, before it is added to the sub-layer input and normalized; also applied to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, $P_{drop} = 0.1$. (The English→French big model used $P_{drop} = 0.1$ rather than 0.3.)
2. **Label smoothing** — value $\epsilon_{ls} = 0.1$. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.

(For inference, the paper reports averaging the last 5 checkpoints for base models — checkpoints written at 10-minute intervals — and the last 20 checkpoints for big models, with beam search using beam size 4 and length penalty $\alpha = 0.6$; maximum output length was set to input length + 50, terminating early when possible.)

---

## Results (paper §6)

### Machine translation (§6.1)

On the WMT 2014 newstest2014 test sets:

- **English→German:** the **big** model reaches **28.4 BLEU**. The paper states it "outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4", trained in 3.5 days on 8 P100 GPUs. Even the **base** model — **27.3 BLEU** — surpasses all previously published models and ensembles (the strongest prior ensemble in the paper's table is ConvS2S at 26.36; the strongest prior single models are MoE 26.03, ConvS2S 25.16, GNMT+RL 24.6) at a fraction of the training cost of any of the competitive models.
- **English→French:** the **big** model sets a **new single-model state-of-the-art BLEU score of 41.8** (abstract and Table 1; the body text of §6.1 quotes 41.0 for the English→French big model, which used dropout 0.1 instead of 0.3). The **base** model reaches **38.1 BLEU**. Prior published numbers in the paper's table: MoE 40.56, ConvS2S 40.46, GNMT+RL 39.92, with the best prior ensemble at 41.29. The paper reports outperforming all previously published single models at less than 1/4 the training cost of the previous state-of-the-art model.
- **Training cost (FLOPs):** Transformer (base) $3.3 \cdot 10^{18}$; Transformer (big) $2.3 \cdot 10^{19}$ — versus, e.g., $7.7 \cdot 10^{19}$ for the ConvS2S ensemble on English→German and $1.1 \cdot 10^{21}$ for the GNMT+RL ensemble on English→French.
- **Parameter counts:** the base model has **65M parameters**; the big model has **213M parameters**.

The paper also demonstrates generalization on English constituency parsing (§6.3) with both large and limited training data.

### Base vs. big model (from the paper's Table 2)

| Model | $N$ | $d_{\text{model}}$ | $d_{ff}$ | $h$ | $d_k$ | $d_v$ | $P_{drop}$ | $\epsilon_{ls}$ | train steps | params |
|---|---|---|---|---|---|---|---|---|---|---|
| **base** | 6 | 512 | 2048 | 8 | 64 | 64 | 0.1 | 0.1 | 100K | 65M |
| **big** | 6 | 1024 | 4096 | 16 | 64 | 64 | 0.3 | 0.1 | 300K | 213M |

Per the paper's table caption, unlisted values are identical to those of the base model, so the big model inherits $d_k = d_v = 64$ and $\epsilon_{ls} = 0.1$. (Development-set figures on newstest2013 English→German: base PPL 4.92 / BLEU 25.8; big PPL 4.33 / BLEU 26.4.) The big model for English→French used $P_{drop} = 0.1$ instead of 0.3.

---

## Takeaways: why self-attention displaced recurrence and convolutions

1. **Constant path length between any two positions.** A self-attention layer connects all positions with a constant number of sequentially executed operations, giving $O(1)$ maximum path length versus $O(n)$ for recurrence and $O(\log_k n)$ for convolution. Because learning long-range dependencies gets harder as forward/backward signal paths get longer, self-attention makes those dependencies dramatically easier to learn.
2. **Parallelism, not just asymptotics.** Recurrent computation is inherently sequential along positions — which the Introduction calls out as precluding parallelization *within* a training example, a constraint that bites at longer sequence lengths. Self-attention needs only $O(1)$ sequential operations per layer, so training parallelizes across positions; this is the concrete reason the base model trains in 12 hours and the big model in 3.5 days on eight GPUs.
3. **Favorable cost at realistic sequence lengths.** Per-layer complexity is $O(n^2 \cdot d)$ for self-attention versus $O(n \cdot d^2)$ for recurrence — self-attention is cheaper whenever $n < d$, which holds for the word-piece / byte-pair sentence representations used by state-of-the-art translation models. (For very long sequences, restricted self-attention over a neighborhood of size $r$ is floated as future work, at the price of $O(n/r)$ path length.)
4. **Convolutions can't connect everything in one layer.** A single convolutional layer with kernel width $k < n$ leaves position pairs unconnected; connecting them takes $O(n/k)$ contiguous or $O(\log_k n)$ dilated layers, lengthening the longest paths. Even a separable convolution at $k = n$ costs exactly a self-attention layer plus a point-wise feed-forward layer — which is precisely the pair of blocks the Transformer stacks, obtained more simply.
5. **The encoder–decoder bottleneck is handled by attention itself.** Where prior systems bolted an attention mechanism between a recurrent encoder and decoder, here attention *is* the encoder and the decoder, with encoder–decoder attention layers carrying the cross-sequence coupling.
6. **A bonus: interpretability.** Attention distributions can be inspected, and the paper shows individual heads learn distinct tasks, many related to syntactic and semantic structure.

In short: self-attention matched or beat the connectivity of recurrence and convolution while being cheaper at typical sequence lengths, strictly more parallelizable, and shorter-path — and the translation results (28.4 EN→DE, 41.8 EN→FR at a fraction of prior training cost) were the empirical proof that "attention is all you need."

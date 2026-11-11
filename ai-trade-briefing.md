# The Paper That Started the AI Trade — Thematic Briefing

**Working title:** 'the paper that started the AI trade'
**Status:** Draft for research meeting circulation. All company facts below are as reported by Yahoo Finance market data (not rounded or converted). All paper facts are quoted from arXiv:1706.03762 ("Attention Is All You Need").

---

## 1. Company Snapshot — NVIDIA Corporation (NVDA)

| Field | Value (as reported by Yahoo Finance) |
| --- | --- |
| Sector / Industry | Technology / Semiconductors |
| Market cap | 4078186790912 USD |
| Current price | 168.89 USD (NasdaqGS, market state: PRE) |
| 52-week range | 164.27 – 243.37 USD (all-time high: 243.37) |
| Analysts | 57 opinions; consensus: strong_buy (recommendationKey); mean target: 325.7772; median: 315.0; range: 180.0 – 515.0 |
| Trailing P/E | 21.351454 |
| Forward P/E | 10.6033 |
| Beta | 2.247 |
| Revenue (TTM) | 302970011648 USD |
| Gross margin | 0.74674004 |
| Return on equity | 1.17211 |
| Employees | 42000 full-time |
| HQ | Santa Clara, California, United States |
| CEO | Mr. Jen-Hsun Huang (Co-Founder, CEO & Director, born 1963) |
| CFO | Ms. Colette M. Kress (Executive VP & CFO) |
| Business | Data-center scale AI infrastructure company; Compute & Networking segment (data center accelerated computing, AI solutions, networking) and Graphics segment (GeForce/Quadro/RTX GPUs) |

---

## 2. Institutional Ownership Table — Top 3 Holders (Per Yahoo Finance Institutional Holders)

All figures exactly as returned by the data service (no rounding or conversion).

| Holder | Shares (as reported) | pctHeld (as reported) | Value (USD, as reported) | pctChange (as reported) | Date Reported (epoch) |
| --- | --- | --- | --- | --- | --- |
| Blackrock Inc. | 1941918386 | 0.0806 | 327970595026 | 0.0085 | 1782777600000 |
| Vanguard Capital Management LLC | 1540816358 | 0.0639 | 260228473762 | 0.0015 | 1782777600000 |
| FMR, LLC | 1026051548 | 0.042600002 | 173289845315 | 0.0324 | 1782777600000 |

**Ownership context (major holders, as reported):**
- insidersPercentHeld: 0.04006
- institutionsPercentHeld: 0.71384 (71.384% of shares outstanding)
- institutionsFloatPercentHeld: 0.74364 (74.364% of float)
- institutionsCount: 7753

---

## 3. Paper Facts — "Attention Is All You Need" (arXiv:1706.03762)

### 3.1 Headline Machine-Translation Results (from the abstract)
- **28.4 BLEU** on the WMT 2014 **English-to-German** translation task — improving over the existing best results, including ensembles, by **over 2 BLEU**.
- **41.8 BLEU** on the WMT 2014 **English-to-French** translation task — a new **single-model state-of-the-art**, achieved after training for **3.5 days on eight GPUs**, a small fraction of the training costs of the best models from the literature.
- Generalization: the Transformer generalizes well to other tasks, applied successfully to English constituency parsing with both large and limited training data.

### 3.2 Encoder / Decoder Layers (base model)
- **Encoder:** a stack of **N = 6 identical layers**, each with two sub-layers (multi-head self-attention; position-wise fully connected feed-forward network), residual connections + layer normalization, with all sub-layers and embeddings producing outputs of dimension **d_model = 512**.
- **Decoder:** also a stack of **N = 6 identical layers**, each with three sub-layers (masked multi-head self-attention; multi-head attention over the encoder output; position-wise feed-forward network), plus residual connections and layer normalization.

### 3.3 Scaled Dot-Product Attention (as described in the paper)
- The attention input consists of queries and keys of dimension **d_k**, and values of dimension **d_v**.
- The dot products of the query with all keys are **divided by √d_k** and a softmax is applied to obtain the weights on the values.
- Formulation: **Attention(Q, K, V) = softmax(QK^T / √d_k) V**
- Scaling rationale: for large values of d_k, the dot products grow large in magnitude, pushing the softmax into regions where it has extremely small gradients; to counteract this, the dot products are scaled by **1/√d_k**. (avoids very large dot products pushing softmax into regions with minuscule gradients)

### 3.4 Multi-Head Attention (as described in the paper)
- Instead of performing a single attention function with d_model-dimensional keys, values and queries, queries, keys and values are **linearly projected h times with different, learned linear projections** to d_k, d_k and d_v dimensions respectively; attention is performed in parallel on each projected version; the d_v-dimensional outputs are **concatenated and once again projected**.
- MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O, where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
- **The base model uses h = 8 parallel attention layers (heads)**, with d_k = d_v = d_model / h = **64**. Due to the reduced dimension of each head, total computational cost is similar to single-head attention with full dimensionality.
- Multi-head attention is used in three ways: encoder-decoder attention, encoder self-attention, and decoder self-attention (with masking to preserve the auto-regressive property).

---

## 4. Sources

- Company profile, market data and institutional ownership: Yahoo Finance (ticker NVDA) — null.
- Paper: Vaswani et al., "Attention Is All You Need", arXiv:1706.03762 — https://arxiv.org/abs/1706.03762 (abstract and Model Architecture section, incl. 3.2 Encoder and Decoder Stacks, 3.2.1 Scaled Dot-Product Attention, 3.2.2 Multi-Head Attention).

---

## 5. Knowledge base records

The following entities and relations were created in the team knowledge graph (verified by read-back).

### Entities (6)

| Entity name | Type | Key observations |
| --- | --- | --- |
| NVIDIA | Company | Sector: Technology; Industry: Semiconductors; market cap 4078186790912 USD; institutional ownership 0.71384 (71.384% of shares); 7753 institutions; HQ Santa Clara, CA; ~42000 employees; beta 2.247; trailing P/E 21.351454; analyst consensus strong_buy (57 opinions); 52-week range 164.27–243.37 |
| Attention Is All You Need | Academic Paper | arXiv 1706.03762; proposes the Transformer based solely on attention, dispensing with recurrence and convolutions; 28.4 BLEU on WMT 2014 En-De (+2 BLEU over prior best incl. ensembles); 41.8 BLEU on WMT 2014 En-Fr (single-model SOTA, 3.5 days on 8 GPUs); generalizes to English constituency parsing; N=6 encoder and decoder layers; scaled dot-product attention softmax(QK^T/√d_k)V; h=8 parallel heads with d_k=d_v=d_model/h=64 |
| Transformer | Architecture | Encoder stack N=6 identical layers (2 sub-layers each); decoder stack N=6 identical layers (3 sub-layers each); scaled dot-product attention softmax(QK^T/√d_k)V; multi-head attention with h=8 heads, d_k=d_v=64; sinusoidal positional encodings; d_model=512, d_ff=2048 |
| Blackrock Inc. | Institutional Holder | Top institutional holder of NVIDIA; most recently reported shares 1941918386; pctHeld 0.0806; value 327970595026 USD; pctChange 0.0085; date reported 2026-06-30 (epoch 1782777600000) |
| Vanguard Capital Management LLC | Institutional Holder | Top institutional holder of NVIDIA; most recently reported shares 1540816358; pctHeld 0.0639; value 260228473762 USD; pctChange 0.0015; date reported 2026-06-30 (epoch 1782777600000) |
| FMR, LLC | Institutional Holder | Top institutional holder of NVIDIA; most recently reported shares 1026051548; pctHeld 0.042600002; value 173289845315 USD; pctChange 0.0324; date reported 2026-06-30 (epoch 1782777600000) |

### Relations (5)

| From | Relation | To |
| --- | --- | --- |
| Blackrock Inc. | is a top institutional holder of | NVIDIA |
| Vanguard Capital Management LLC | is a top institutional holder of | NVIDIA |
| FMR, LLC | is a top institutional holder of | NVIDIA |
| Attention Is All You Need | introduces | Transformer |
| Transformer | underpins | NVIDIA's data-center AI business |

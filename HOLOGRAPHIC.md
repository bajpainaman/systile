# The Holographic Tensor Store

A `HoloMemory` is a key→value associative store with one unusual property: it does
not have slots. Every entry is **summed on top of every other entry** inside a
single fixed-width vector, and lookup is recovered by algebra plus one matrix
multiply. This document explains the mechanism, the capacity, why it is shaped for
a TPU, and — honestly — what is and isn't new about it.

## The mechanism

Pick a dimension `d` (say 8192). Every symbol — every key, every value — is a
deterministic **bipolar hypervector** in `{-1, +1}^d`, drawn so that two distinct
symbols are almost orthogonal (their cosine is `~ N(0, 1/d)`). Three operations:

| Operation | Definition | Meaning |
| --- | --- | --- |
| bind `⊛` | elementwise product | attach a value to a key; result is dissimilar to both |
| bundle `+` | elementwise sum | superpose many vectors into one similar to each |
| similarity | dot product | how much two hypervectors agree |

Bipolar bind is **self-inverse**: `x ⊛ x = 1`, the all-ones vector. That single fact
is what makes the structure work.

**Insert** binds the key to the value and bundles it into the memory:

```text
memory  +=  atom(key) ⊛ atom(value)
```

**Lookup** unbinds the key — which cancels it back out — and cleans up what remains:

```text
noisy = memory ⊛ atom(key)
      = atom(value_k)  +  Σ_{i≠k} atom(key_i) ⊛ atom(value_i) ⊛ atom(key_k)
      = atom(value_k)  +  (near-orthogonal noise)
value = argmax_j  noisy · codebook[:, j]
```

The first term is the wanted value, recovered cleanly because the key cancels
itself. Every other entry survives as a product of three near-random bipolar
vectors, i.e. near-orthogonal noise with mean zero. The **cleanup** step projects
the noisy result onto the value codebook and takes the best match.

## Why the cleanup is a matmul

The value codebook is a `d × m` matrix `C` whose columns are the `m` value atoms.
Cleaning up a single probe is the matrix–vector product `Cᵀ·noisy` followed by an
argmax. Cleaning up a **batch** of `b` probes at once is the matrix–matrix product

```text
S = Q · C          Q is b × d,   C is d × m,   S is b × m
```

after which each row's argmax names the recovered value. That product is the whole
point: in `systile` it runs through the systolic matmul engine
([`Codebook::cleanup_batch`]), so a single MXU-shaped GEMM resolves the entire batch
of lookups against the entire vocabulary at once.

## Capacity

Crosstalk analysis gives a signal-to-noise ratio of `√(d/K)` for `K` bundled pairs,
and beating the maximum of `~m` near-Gaussian distractors in the argmax needs to
clear `√(2 ln m)`. Setting them equal yields the standard rule of thumb:

```text
K_max  ≈  d / (2 ln m)
```

linear in the dimension, logarithmic in the vocabulary size. `systile` exposes this
as [`HoloMemory::estimated_capacity`] and [`HoloMemory::load_factor`]. The
`holo_capacity` example measures the real curve: recall stays near 100% below
`K_max` and degrades gracefully above it, exactly as predicted. The information
capacity of the representation is roughly `0.36–0.5` bits per dimension (Frady,
Kleyko & Sommer 2018).

## Why this is a TPU-shaped data structure (and the honest caveat)

On a CPU this is a *bad* map: a hash table is `O(1)` and exact, while this is a dense
`O(m·d)` matmul that only *probably* returns the right value. The trade is only
worth it where dense matmul is the cheap primitive and branch-y pointer chasing is
expensive — a TPU/tensor-core — and where you want to (a) resolve thousands of
lookups in one batched GEMM, (b) tolerate noise, or (c) keep the whole structure
differentiable.

To be precise and not overclaim: this is **not impossible on a CPU or GPU** — the
cleanup is an ordinary GEMM and runs anywhere. It is that nowhere else is it the
*right* shape. The honest framing is "matmul-native / maps efficiently onto the TPU
MXU," not "TPU-exclusive." And it is a **bounded-capacity, error-rate-tunable**
associative memory, fundamentally approximate — beyond `K_max` it returns wrong
values silently. It trades exact, bandwidth-bound random access for approximate,
compute-bound dense matmul. That trade is the entire pitch.

## What is new here, and what isn't

The algebra is established and the lineage is deep — claiming otherwise would be
dishonest:

- **MAP** (Multiply-Add-Permute), bipolar bind/bundle: Gayler 2003.
- **HRR** (the convolution variant) and the capacity analysis: Plate 1995/2003.
- **Binary spatter codes** and the record/holistic mapping: Kanerva 2009.
- **Capacity as bits/dimension**: Frady, Kleyko & Sommer 2018; Kleyko et al. survey
  2022/2023.
- **Cleanup / item memory as argmax over a codebook**, and the iterative
  factorization for unknown factors: **Resonator Networks**, Frady et al. 2020.
- **Key-value-in-superposition with matmul cleanup** specifically: Liu et al. 2025
  (Kronecker-rotation cleanup) — the closest direct prior art.
- **Algorithms as linear algebra** generally: GraphBLAS (Kepner & Gilbert 2011;
  Davis 2019); Tensor Product Representations (Smolensky 1990); differentiable
  memories (Neural Turing Machines, Graves et al. 2014).

What `systile` contributes is **packaging and systems framing**, not new
mathematics: a concrete, training-free Rust associative-array data structure whose
storage is one fixed-width tensor, whose dominant operation is a single dense GEMM
deliberately tiled for the systolic MXU (reusing the same `Geometry`/`Bf16`/matmul
substrate as the rest of the crate), with capacity and load exposed through an
ordinary map API.

## References

- Gayler, "Vector Symbolic Architectures answer Jackendoff's challenges," 2003 — arXiv:cs/0412059
- Plate, *Holographic Reduced Representations*, 1995/2003
- Kanerva, "Hyperdimensional Computing," *Cognitive Computation* 1, 2009
- Schlegel, Neubert & Protzel, "A comparison of VSAs," 2022 — arXiv:2001.11797
- Kleyko et al., "A Survey on HDC/VSA," 2022/2023 — arXiv:2111.06077 / arXiv:2112.15424
- Frady, Kleyko & Sommer, "A Theory of Sequence Indexing…," *Neural Computation*, 2018 — arXiv:1803.00412
- Frady, Kent, Olshausen & Sommer, "Resonator Networks," *Neural Computation*, 2020 — arXiv:2007.03748
- Liu et al., "Linearithmic Clean-up for Vector-Symbolic Key-Value Memory…," 2025 — arXiv:2506.15793
- Kepner & Gilbert, *Graphs in the Language of Linear Algebra*, SIAM, 2011; Davis, "SuiteSparse:GraphBLAS," *ACM TOMS*, 2019
- Smolensky, "Tensor product variable binding," *Artificial Intelligence*, 1990; Graves et al., "Neural Turing Machines," 2014 — arXiv:1410.5401

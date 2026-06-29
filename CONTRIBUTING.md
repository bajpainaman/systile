# Contributing to systile

Thanks for your interest! `systile` is a small, focused crate and contributions are
welcome.

## Ground rules

- The crate is `#![forbid(unsafe_code)]`. Keep it that way; reach for a safe
  abstraction instead.
- Every public item needs a doc comment — `#![warn(missing_docs)]` is on.
- New behaviour needs a test. Correctness-sensitive code (anything touching the
  layout math or the matmul simulator) should be checked against an independent
  reference, not just against itself.

## Before you open a PR

```
cargo fmt --all
cargo clippy --all-targets --all-features
cargo test --all-features
```

CI runs all three with `-D warnings`, plus an MSRV build on Rust 1.74.

## Commit style

Commits follow [Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`, `bench:`. Keep them small
and atomic — one logical change each.

## Design notes

The guiding principle is that the data structure mirrors TPU hardware constraints.
If a change makes the layout *less* faithful to how a real systolic accelerator
addresses memory, it probably belongs behind a feature flag or in a separate type.

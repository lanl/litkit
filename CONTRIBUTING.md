# Contributing to LitKit

Thanks for your interest in improving LitKit. Bug reports, questions, and pull
requests are welcome on [GitHub](https://github.com/lanl/litkit/issues).

## Certificate of origin

Contributions are accepted under the project's [MIT license](LICENSE), the same
terms LitKit is distributed under.

Please sign off on your commits, certifying that you wrote the contribution or
otherwise have the right to submit it under that license — the
[Developer Certificate of Origin](https://developercertificate.org). Add `-s`
when you commit:

```sh
git commit -s -m "your message"
```

which appends a `Signed-off-by:` line using the name and email from your
`git config`.

## Development setup

LitKit uses [uv](https://docs.astral.sh/uv/) and supports Python 3.12 on macOS
(Apple Silicon) and Linux (aarch64). Install LitKit with the development tools
(ruff, black, mypy, pytest):

```sh
git clone https://github.com/lanl/litkit.git
cd litkit
uv sync
```

The development tools are in the `dev` dependency group, which `uv sync` and
`uv run` install by default. `uv sync --no-dev` leaves them out.

If you change dependencies in `pyproject.toml`, run `uv lock` and commit
`uv.lock` in the same pull request. `uv lock --check` confirms the lock file is
current.

## Checks

```sh
uv run ruff check src/
uv run black --check src/
```

Both are configured in `pyproject.toml`. Older code doesn't pass yet, so keep
your changes clean and don't reformat files you aren't otherwise changing:
unrelated churn makes a pull request hard to review.

## Testing

```sh
uv run pytest
```

runs the default suite: a few seconds, with no GPU, network access, model files
or corpus data. Every test gets a fresh workspace under a temporary directory,
with `LITKIT_*`, `HF_*` and `OPENAI_*` environment variables cleared, so the
suite never touches a real workspace.

Tests that need more are marked and excluded by default:

| Marker | Needs | Run with |
|---|---|---|
| `slow` | real models | `LITKIT_TEST_HF_HOME=/path/to/hf_cache uv run pytest -m slow` |
| `gpu` | real models and a CUDA or Apple MPS device | `LITKIT_TEST_HF_HOME=/path/to/hf_cache uv run pytest -m gpu` |
| `network` | network access or an API key | none yet |

`LITKIT_TEST_HF_HOME=/path/to/hf_cache uv run pytest -m ""` runs everything. The `slow` tests load SPECTER2 and
`all-mpnet-base-v2` from `LITKIT_TEST_HF_HOME`, a Hugging Face cache that holds
`hub/models--allenai--specter2_base` and
`hub/models--sentence-transformers--all-mpnet-base-v2`, and skip when it isn't
set, as they do where torch isn't installed. Markers are strict: a misspelled
marker is an error.

`tests/conftest.py` has the shared pieces:

- `FakeEmbedder`, with the same `encode()` contract as the real embedders.
  It returns bag-of-words vectors, so texts that share words score higher, and
  a retrieval test can have a known right answer without a model.
- `jats()` and `pubmed_xml()`, which build small article records, and
  `tiny_corpus`, two tars with edge cases (an empty `<article>`, a missing
  abstract, a PubMed record, a non-XML member, the same PMCID in both tars).
- `make_index`, `unit_vectors` and a seeded `rng` for FAISS tests.

`tests/integration/` runs the real CLI in a subprocess with the fake embedders
(`run_litkit_fake.py`): a build of the tiny corpus, `--update`, and a query.

A test for a known, unfixed bug is marked
`@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#N: ...")` with
the issue number. It then fails as soon as the bug is fixed: remove the marker in
the fix's pull request. `raises=AssertionError` keeps an unrelated crash from
counting as the expected failure, so inside such a test, check for anything other
than the bug itself without `assert` (the integration tests' `_ok()` raises
`CliFailed` when the CLI exits nonzero, and `_check()` raises `WrongResult`). A bug fix needs a test that fails
without the fix.

To test a change end to end with real models, build a small index and validate it:

```sh
uv run ./test_build.sh --clean
```

This builds from the tar files listed in `workspace/mac_test.manifest` and then
runs `validate_build.sh` on the result. The tar files aren't part of the
repository: edit the manifest to list absolute paths to a few tar files of
JATS/NXML articles, such as some from the
[PMC Open Access Subset](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/).

If you change multi-node builds, run a multi-node build too
(`vector_build_multi.sbatch`; see [LITKIT_CLUSTER_GUIDE.md](LITKIT_CLUSTER_GUIDE.md)).

## Concurrency

LitKit allows exactly one writer per SQLite database and FAISS index set. In a
multi-node build, each producer (`--embed-producer`) writes its own shard
database and segment files, and a single consumer (`--consume-only --faiss-writer`)
merges them. Don't add code paths in which several processes write to the main
database: SQLite writers on a network filesystem fail with `SQLITE_BUSY` or
corrupt data.

## Changelog

For a change a user would notice, add an entry under `[Unreleased]` in
[CHANGELOG.md](CHANGELOG.md). Use one or two lines that say what changed and what
users need to do differently, if anything. Refactors, internal fixes, and
documentation changes don't need an entry.

## Releasing

To bump the version, update the version string in all of these:

- `pyproject.toml` (`version = "X.Y.Z"`), then run `uv lock` to update `uv.lock`
- `justfile` (`tag := "vX.Y.Z-" ...`)
- the `IMG=` lines in `vector_build_single.sbatch`, `vector_build_multi.sbatch`,
  and `vector_resume_consumer.sbatch`
- the version in `LITKIT_CLUSTER_GUIDE.md` and `LITKIT_MAC_GUIDE.md`

`git grep -n "<old version>"` finds any you missed. Then move the
`[Unreleased]` entries in `CHANGELOG.md` under a new version heading, commit, and
tag the commit `vX.Y.Z`.

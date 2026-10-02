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
uv sync --extra dev
```

`uv sync` makes the environment match exactly the extras you name, so a later
bare `uv sync` removes the development tools. Pass `--extra dev` every time.

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

LitKit doesn't have a unit test suite yet; contributions toward one are
especially welcome. Use pytest, and put tests under `tests/`. Tests should run in
seconds without a GPU, network access, model downloads, or real corpus data.

To test a change end to end, build a small index and validate it:

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

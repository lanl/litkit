# Changelog

Notable changes to LitKit, in [Keep a Changelog](https://keepachangelog.com) format.
Versions before the open-source release were internal and were not tagged; earlier
internal versions are summarized under 0.3.33.

## [Unreleased]

### Added
- Open-source release under the MIT license (LANL O5068).
- macOS (Apple Silicon) support in `uv.lock`, so `uv sync` installs on a Mac; `install_mac.sh`.

### Changed
- macOS requires torch ≥ 2.6, which fixes CVE-2025-32434; Linux stays on 2.5.1.

### Fixed
- Builds slowed down as the index grew, because every batch scanned the whole FAISS
  index twice to remove IDs. `--rebuild` now skips the scan, and updates scan once.

## [0.3.35] - 2025-12-24

### Added
- `--strict-ingest`. By default, an XML member that fails to ingest is now skipped
  and the build continues, instead of being retried on every run.
- Batch scripts stage producer output on node-local SSD (`USE_LOCAL_STAGING`) and
  cancel the whole job when any producer fails.

### Changed
- Multi-node builds: each producer now writes its own SQLite shard database, which
  the consumer merges. Segment files use content-derived IDs, so they stay valid
  after the merge.
- `cli.py` split into `build`, `concurrent`, `config`, `db`, `index`, `llm`,
  `pipeline`, `retrieval`, and `segments` packages; LLM calls go through a
  provider-agnostic `litkit.llm` module.
- The writer guard no longer expires after 24 hours by default, so a long build is
  never evicted while it is still running. Set `LITKIT_WRITER_GUARD_TTL` (in
  seconds) to restore automatic expiry.

### Fixed
- Citations in answers could point to the wrong papers after a context-overflow
  retry dropped chunks.
- `--parse-workers` had no effect; uncompressed tar files are now parsed in parallel.
- macOS: crash at startup with FAISS + MPS, and on `faiss-cpu` builds without
  `faiss.cvar.seed`.

## [0.3.34] - 2025-12-12

### Added
- `doc_id` column on `papers` (PMC ID, then PMID, then a content hash) for
  deduplication across producers.
- Multi-node build options `--embed-producer`, `--consume-only`, and
  `--init-indices-only`, with tar files spread across shards by size.
- Charliecloud container builds (`Dockerfile.lean`, `justfile`) and Slurm batch
  scripts.
- `.nxml` files are ingested alongside `.xml`.

### Fixed
- With several GPUs, a fast-starting worker could take all the work while the others
  sat idle.

## [0.3.33] - 2025-10-27

### Added
- Package layout under `src/litkit`, with the `litkit` command-line entry point.
- Embedding backends: SPECTER2 for papers, SBERT MPNet for chunks, and a
  multi-device embedding pool.

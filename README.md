# litSearch

Automated literature discovery and ranking pipeline for research papers. Given a set of seed papers, litSearch discovers recent related work via the Semantic Scholar API and ranks candidates using semantic similarity and metadata heuristics.

## How it works

1. **Seed loading** — Reads arXiv URLs/IDs (and DOI/publisher URLs) from a text file.
2. **Discovery** — Finds recent papers through three channels:
   - **Keyword search** — Bulk search using weighted keywords from `config.yaml`.
   - **Citation crawl** — Papers that cite any of your seed papers.
   - **Author crawl** — Recent papers by high-impact authors (h-index > 50) from your seeds.
3. **Enrichment** — Batch-fetches author metadata (h-index, affiliations) for candidates.
4. **Semantic embedding** — Encodes seed and candidate abstracts with `all-MiniLM-L6-v2`, computes cosine similarity. Previously seen embeddings are cached in `cache/`.
5. **Scoring** — Ranks candidates across 8 heuristics with auto-calibrated thresholds.
6. **Output** — Writes a ranked report to a text file.

## Scoring heuristics

All point values are configurable in `config.yaml` under the `scoring:` section.

| Category | Heuristic | Default max | Description |
|----------|-----------|-------------|-------------|
| Relevance | Semantic similarity | 40 | Cosine similarity to seed papers (threshold auto-calibrated from inter-seed similarity) |
| Relevance | Author in seed refs | 10 | 4 pts per candidate author found in seed reference lists (matched by SS authorId) |
| Relevance | Cites seed | 10 | 5 pts per seed paper directly cited (detected via citation crawl) |
| Quality | Author authority | 10 | Log-scaled h-index of top author |
| Quality | Citation velocity | 10 | Citations/day, with recency prior scaled to window size |
| Bonus | Institutional authority | 5 | Author affiliated with a known AI research lab (word-boundary matching) |
| Bonus | Category intersection | 5 | Paper spans cs.DB and cs.LG/stat.ML |
| Bonus | Benchmark specificity | 5 | Named benchmark datasets in abstract |

## Usage

```bash
# Basic: discover papers from the last 7 days
python main.py --input ref_papers.txt --days 7 --output results.txt

# Faster run without author/reference enrichment
python main.py --input ref_papers.txt --days 14 --no-enrich

# With API key for higher rate limits
python main.py --input ref_papers.txt --days 7 --api-key YOUR_KEY
# or
export SS_API_KEY=YOUR_KEY
python main.py --input ref_papers.txt --days 7
```

### Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `--input`, `-i` | Text file with arXiv URLs/IDs (one per line) | *required* |
| `--days`, `-t` | Lookback window in days | `7` |
| `--output`, `-o` | Output file path | `results.txt` |
| `--no-enrich` | Skip author/reference enrichment (faster, degrades 3 heuristics) | off |
| `--api-key` | Semantic Scholar API key | `$SS_API_KEY` |

### Seed file format

One URL or ID per line. Lines starting with `#` are ignored.

```text
https://arxiv.org/abs/2505.10960
https://arxiv.org/pdf/2510.06377
https://www.nature.com/articles/s41586-024-08328-6
# commented-out papers are skipped
arxiv:2502.17361
```

## Configuration

`config.yaml` controls three things:

**Keyword weights** — drive the Semantic Scholar search query:
```yaml
keyword_weights:
  relational foundation model: 5
  tabular transformer: 5
  graph transformer: 3
```

**Scoring constants** — tune point allocations without code changes:
```yaml
scoring:
  max_semantic: 40
  max_cites_seed: 10
  cites_seed_per_seed: 5
  # ... see config.yaml for all options
```

**Institutions and benchmarks** — customize the lists used by those heuristics:
```yaml
institutions:
  - Stanford
  - DeepMind
  # ...
benchmarks:
  - RelBench
  - OGB
  # ...
```

## Setup

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- `requests`, `numpy`, `pyyaml`, `sentence-transformers`
- (Optional) Semantic Scholar API key for higher rate limits

## Tests

```bash
python -m pytest tests/ -v
```

## Project structure

```
litSearch/
  main.py          # CLI entrypoint and report formatting
  fetcher.py       # Semantic Scholar API client and paper discovery
  scorer.py        # Semantic similarity + metadata scoring engine
  config.yaml      # Keywords, scoring constants, institutions, benchmarks
  ref_papers.txt   # Seed paper URLs
  requirements.txt
  cache/           # Embedding cache (auto-created)
  tests/
    conftest.py        # Path setup for imports
    test_scorer.py     # Scorer unit tests
    test_fetcher.py    # Fetcher/URL parsing unit tests
```

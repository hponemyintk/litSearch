# litSearch

Automated literature discovery and ranking pipeline for research papers. Given a set of seed papers, litSearch discovers recent related work via the Semantic Scholar API and ranks candidates using a multi-heuristic scoring system.

## How it works

1. **Seed loading** — Reads arXiv URLs/IDs (and DOI/publisher URLs) from a text file.
2. **Discovery** — Finds recent papers through three channels:
   - **Keyword search** — Bulk search using weighted keywords from `config.yaml`.
   - **Citation crawl** — Papers that cite any of your seed papers.
   - **Author crawl** — Recent papers by high-impact authors (h-index > 50) from your seeds.
3. **Enrichment** — Batch-fetches reference lists and author metadata (h-index, affiliations).
4. **Scoring** — Ranks candidates on a 100-point scale across 9 heuristics.
5. **Output** — Writes a ranked report to a text file.

## Scoring heuristics

| Category | Heuristic | Max pts |
|----------|-----------|---------|
| Relevance | Weighted keyword match (title 2x, abstract 1x) | 30 |
| Relevance | Reference overlap (shared authors/topics with seeds) | 15 |
| Relevance | Keyword coverage (fraction of config keywords hit) | 10 |
| Relevance | Author in seed references (community membership) | 10 |
| Quality | Author authority (log-scaled h-index) | 10 |
| Quality | Citation velocity (effective citations/day) | 10 |
| Bonus | Institutional authority (target lab affiliations) | 5 |
| Bonus | Category intersection (cs.DB + cs.LG/stat.ML) | 5 |
| Bonus | Benchmark specificity (named tools/datasets) | 5 |

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
| `--no-enrich` | Skip author/reference enrichment (faster) | off |
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

Edit `config.yaml` to customize keyword weights. Higher weights increase a keyword's contribution to the relevance score. These keywords also drive the Semantic Scholar search query.

```yaml
keyword_weights:
  relational foundation model: 5
  tabular transformer: 5
  graph transformer: 3
  transformer: 2
```

## Setup

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- `requests`, `scikit-learn`, `numpy`, `pyyaml`
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
  scorer.py        # 9-heuristic scoring engine
  config.yaml      # Keyword weights configuration
  ref_papers.txt   # Seed paper URLs
  requirements.txt
  tests/
    conftest.py        # Path setup for imports
    test_scorer.py     # Scorer unit tests
    test_fetcher.py    # Fetcher/URL parsing unit tests
```

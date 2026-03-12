"""
main.py — CLI entrypoint for the daily paper discovery pipeline.

Usage:
    python main.py --input papers.txt --days 7 --output results.txt
    python main.py --input papers.txt --days 14 --no-enrich
"""

import argparse
import logging
import sys
from pathlib import Path

from fetcher import (
    load_seed_urls, fetch_seed_papers, fetch_recent_papers,
    enrich_with_ss, configure_api_key,
)
from scorer import build_scoring_config, rank_papers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def format_paper(rank: int, paper: dict) -> str:
    """Format a single ranked paper as a human-readable text block."""
    bd = paper.get("score_breakdown", {})
    lines = [
        f"{'='*70}",
        f"#{rank}  {paper.get('title', 'Untitled')}",
        f"arXiv: {paper.get('arxiv_id', 'N/A')}",
        f"Published: {paper.get('published', 'unknown')}",
        f"Authors: {', '.join(a.get('name','?') for a in (paper.get('authors') or [])[:5])}",
        f"Categories: {', '.join(paper.get('categories') or [])}",
        f"",
        f"SCORE: {bd.get('summary', 'N/A')}",
        f"",
        f"Abstract:",
        (paper.get("abstract") or "N/A")[:600] + ("..." if len(paper.get("abstract") or "") > 600 else ""),
        f"{'='*70}",
        "",
    ]
    return "\n".join(lines)


def run_pipeline(input_file: str, days: int, output_file: str, enrich: bool) -> None:
    log.info("Loading seed papers from: %s", input_file)
    seed_ids = load_seed_urls(input_file)
    if not seed_ids:
        log.error("No valid arXiv IDs found in %s", input_file)
        sys.exit(1)
    log.info("Loaded %d seed arXiv IDs", len(seed_ids))

    hours = days * 24

    # 1 SS batch call — seeds with references + citations
    log.info("Fetching seed paper metadata from SS...")
    seed_papers = fetch_seed_papers(seed_ids)
    if not seed_papers:
        log.error("Could not fetch any seed paper metadata")
        sys.exit(1)

    # 1-2 SS batch calls — author h-index for seeds
    if enrich:
        log.info("Enriching %d seed papers (author h-index)...", len(seed_papers))
        enrich_with_ss(seed_papers)

    log.info("Building dynamic scoring config from %d seed papers...", len(seed_papers))
    config = build_scoring_config(seed_papers, window_hours=hours)
    log.info(
        "Config: %d keywords, %d target authors, %d ref topics, %d seed-ref authors",
        len(config["keyword_weights"]),
        len(config["target_authors"]),
        len(config["target_topics"]),
        len(config["seed_ref_authors"]),
    )

    # 1-3 SS search calls — discover recent papers
    log.info("Discovering related papers from the last %d days...", days)
    papers = fetch_recent_papers(seed_papers, hours=hours)
    log.info("Found %d candidate papers", len(papers))

    if not papers:
        log.warning("No papers found in the last %d-day window", days)
        Path(output_file).write_text("No papers found.\n", encoding="utf-8")
        return

    # 2-4 SS batch calls — references + author h-index for candidates
    if enrich:
        log.info("Enriching %d candidates (refs + h-index)...", len(papers))
        enrich_with_ss(papers)

    log.info("Scoring and ranking papers...")
    ranked = rank_papers(papers, config)

    out_lines = [
        f"Daily Paper Discovery Report",
        f"Seed papers: {len(seed_ids)} | Candidates scored: {len(ranked)} | Window: last {days}d",
        f"Generated: {__import__('datetime').datetime.now().isoformat()}",
        "",
    ]
    for i, paper in enumerate(ranked, start=1):
        out_lines.append(format_paper(i, paper))

    output = "\n".join(out_lines)
    Path(output_file).write_text(output, encoding="utf-8")
    log.info("Results written to: %s", output_file)

    print(f"\n--- {len(ranked)} Papers (ranked) ---")
    for i, paper in enumerate(ranked, start=1):
        bd = paper.get("score_breakdown", {})
        print(f"  #{i:<3} [{bd.get('total','?'):>3}/100] {paper.get('title','')[:75]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover and score arXiv papers related to your research interests."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to text file containing arXiv URLs/IDs (one per line)."
    )
    parser.add_argument(
        "--days", "-t", type=int, default=7,
        help="Look back window in days (default: 7)."
    )
    parser.add_argument(
        "--output", "-o", default="results.txt",
        help="Output file path (default: results.txt)."
    )
    parser.add_argument(
        "--no-enrich", action="store_true",
        help="Skip author/reference enrichment (faster, less accurate scoring)."
    )
    parser.add_argument(
        "--api-key", default=None,
        help="Semantic Scholar API key (or set SS_API_KEY env var)."
    )
    args = parser.parse_args()
    if args.api_key:
        configure_api_key(args.api_key)
    run_pipeline(args.input, args.days, args.output, enrich=not args.no_enrich)


if __name__ == "__main__":
    main()

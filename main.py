"""
main.py — CLI entrypoint for the daily paper discovery pipeline.

Usage:
    python main.py --input papers.txt --days 7 --output results.txt
    python main.py --input papers.txt --days 14 --no-enrich
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from fetcher import (
    load_seed_urls, fetch_seed_papers, fetch_recent_papers,
    enrich_with_ss, configure_api_key,
)
from scorer import build_scoring_config, precompute_similarities, rank_papers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def format_paper(rank: int, paper: dict, config: dict) -> str:
    """Format a single ranked paper as a human-readable text block."""
    bd = paper.get("score_breakdown", {})
    mt = bd.get("max_total", config.get("max_total", 100))
    lines = [
        f"{'='*70}",
        f"#{rank}  {paper.get('title', 'Untitled')}",
        f"arXiv: {paper.get('arxiv_id', 'N/A')}",
        f"Published: {paper.get('published', 'unknown')}",
        f"Authors: {', '.join(a.get('name','?') for a in (paper.get('authors') or [])[:5])}",
        f"Venue: {paper.get('venue') or 'N/A'}",
        f"Categories: {', '.join(paper.get('categories') or [])}",
        f"",
        f"TOTAL SCORE: {bd.get('total', 0)}/{mt}",
        f"  Semantic Similarity:    {bd.get('semantic', 0):>3}/{config.get('max_semantic', 35)}",
        f"  Seed Author:            {bd.get('seed_ref', 0):>3}/{config.get('max_seed_ref_author', 7)}",
        f"  Cites Seed:             {bd.get('cites_seed', 0):>3}/{config.get('max_cites_seed', 8)}",
        f"  Referenced by Seed:     {bd.get('ref_by_seed', 0):>3}/{config.get('flat_ref_by_seed', 5)}",
        f"  Author Authority:       {bd.get('authority', 0):>3}/{config.get('max_authority', 10)}",
        f"  Citation Velocity:      {bd.get('velocity', 0):>3}/{config.get('max_citation_vel', 20)}",
        f"  Institutional Auth:     {bd.get('institution', 0):>3}/{config.get('flat_institution', 5)}",
        f"  Benchmark Specificity:  {bd.get('benchmark', 0):>3}/{config.get('max_benchmark', 10)}",
        f"  Venue Prestige:         {bd.get('venue', 0):>3}/{config.get('venue_prestige_tier1', 8)}",
        f"",
        f"Abstract:",
        (paper.get("abstract") or "N/A")[:600] + ("..." if len(paper.get("abstract") or "") > 600 else ""),
        f"{'='*70}",
        "",
    ]
    return "\n".join(lines)


def run_pipeline(input_file: str, days: int, output_file: str, enrich: bool,
                  semantic_only: bool = False) -> None:
    wall_start = time.monotonic()
    log.info("Loading seed papers from: %s", input_file)
    seed_ids = load_seed_urls(input_file)
    if not seed_ids:
        log.error("No valid paper IDs found in %s", input_file)
        sys.exit(1)
    log.info("Loaded %d seed paper IDs", len(seed_ids))

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
        "Config: %d seed embeddings, %d seed author IDs, %d seed ref papers, sim threshold=%.3f",
        config["_seed_embeddings"].shape[0],
        len(config["seed_author_ids"]),
        len(config["seed_ref_paper_ids"]),
        config["_sim_threshold"],
    )

    # 1-3 SS search calls — discover recent papers
    log.info("Discovering related papers from the last %d days...", days)
    papers = fetch_recent_papers(seed_papers, hours=hours)
    log.info("Found %d candidate papers", len(papers))

    if not papers:
        elapsed = time.monotonic() - wall_start
        minutes, seconds = divmod(elapsed, 60)
        log.warning("No papers found in the last %d-day window", days)
        log.info("Wall time: %dm %.1fs", int(minutes), seconds)
        Path(output_file).write_text("No papers found.\n", encoding="utf-8")
        return

    # Batch calls — author h-index/affiliations for candidates (skip references)
    if enrich:
        log.info("Enriching %d candidates (h-index + affiliations)...", len(papers))
        enrich_with_ss(papers, skip_references=True)
    else:
        log.warning(
            "--no-enrich: author authority (max %d), institutional (max %d), "
            "and seed-ref matching (max %d) will be degraded — %d/%d pts unreachable",
            config.get("max_authority", 9),
            config.get("flat_institution", 5),
            config.get("max_seed_ref_author", 10),
            config.get("max_authority", 9) + config.get("flat_institution", 5)
            + config.get("max_seed_ref_author", 10),
            config.get("max_total", 100),
        )

    log.info("Computing semantic similarities...")
    precompute_similarities(papers, config)

    mode = "semantic-only" if semantic_only else "full"
    log.info("Scoring and ranking papers (mode=%s)...", mode)
    ranked = rank_papers(papers, config, semantic_only=semantic_only)

    elapsed = time.monotonic() - wall_start
    minutes, seconds = divmod(elapsed, 60)
    wall_str = f"{int(minutes)}m {seconds:.1f}s"

    out_lines = [
        f"Daily Paper Discovery Report",
        f"Seed papers: {len(seed_ids)} | Candidates scored: {len(ranked)} | Window: last {days}d",
        f"Generated: {datetime.now().isoformat()}",
        "",
    ]
    for i, paper in enumerate(ranked, start=1):
        out_lines.append(format_paper(i, paper, config))
    out_lines.append(f"Wall time: {wall_str}")

    output = "\n".join(out_lines)
    Path(output_file).write_text(output, encoding="utf-8")
    log.info("Results written to: %s", output_file)
    log.info("Wall time: %s", wall_str)

    top_n = min(30, len(ranked))
    print(f"\n--- Top {top_n} of {len(ranked)} Papers (ranked) ---")
    for i, paper in enumerate(ranked[:30], start=1):
        bd = paper.get("score_breakdown", {})
        mt = bd.get('max_total', 100)
        print(f"  #{i:<3} [{bd.get('total','?'):>3}/{mt}] {paper.get('title','')[:75]}")
    if len(ranked) > 30:
        print(f"  ... and {len(ranked) - 30} more (see {output_file})")
    print(f"\nWall time: {wall_str}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover and score arXiv papers related to your research interests."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to text file with paper URLs/IDs (arXiv, DOI, Nature, PubMed, etc.)."
    )
    parser.add_argument(
        "--days", "-t", type=int, default=7,
        help="Look back window in days (default: 7)."
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output file path (default: YYYY_MM_dd_Nd_output.txt)."
    )
    parser.add_argument(
        "--no-enrich", action="store_true",
        help="Skip author/reference enrichment (faster, less accurate scoring)."
    )
    parser.add_argument(
        "--semantic-only", action="store_true",
        help="Rank papers purely by semantic similarity to seed papers."
    )
    parser.add_argument(
        "--api-key", default=None,
        help="Semantic Scholar API key (or set SS_API_KEY env var)."
    )
    args = parser.parse_args()
    if args.api_key:
        configure_api_key(args.api_key)
    input_stem = Path(args.input).stem
    suffix = "_semantic" if args.semantic_only else ""
    output = args.output or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{input_stem}_{args.days}d{suffix}_output.txt"
    run_pipeline(args.input, args.days, output, enrich=not args.no_enrich,
                 semantic_only=args.semantic_only)


if __name__ == "__main__":
    main()

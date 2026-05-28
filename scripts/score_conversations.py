#!/usr/bin/env python3
"""
scripts/score_conversations.py
────────────────────────────────
CLI tool to score a folder of conversation JSON files.

Usage:
    python scripts/score_conversations.py \
        --input data/sample_conversations/ \
        --output results/ \
        --facets data/facets_processed.csv \
        --model qwen2.5:3b \
        --facet-ids fluency dishonesty emotionalism   # optional subset
"""

import asyncio
import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ollama_client import OllamaClient
from src.pipeline.facet_registry import FacetRegistry
from src.pipeline.turn_splitter import split_conversation, load_conversation_file
from src.scoring.orchestrator import ScoringOrchestrator
from src.scoring.aggregator import results_to_dict, compute_summary, save_results

app = typer.Typer(help="ConvoEval CLI — score conversations across hundreds of facets")
console = Console()


@app.command()
def score(
    input_dir: Path = typer.Option(..., "--input", "-i", help="Folder with conversation JSON files"),
    output_dir: Path = typer.Option(..., "--output", "-o", help="Output folder for results"),
    facets_csv: Path = typer.Option("data/facets_processed.csv", "--facets", "-f"),
    model: str = typer.Option("qwen2.5:3b", "--model", "-m"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url"),
    facet_ids: list[str] = typer.Option(None, "--facet-ids", help="Subset of facet IDs"),
    score_user_turns: bool = typer.Option(False, "--score-user-turns"),
    chunk_size: int = typer.Option(20, "--chunk-size"),
):
    """Score all conversation files in INPUT_DIR."""
    asyncio.run(_run(
        input_dir, output_dir, facets_csv, model, ollama_url,
        facet_ids or None, score_user_turns, chunk_size
    ))


async def _run(input_dir, output_dir, facets_csv, model, ollama_url,
               facet_ids, score_user_turns, chunk_size):
    # Load registry
    registry = FacetRegistry()
    if not facets_csv.exists():
        console.print(f"[red]Facets CSV not found: {facets_csv}[/red]")
        raise typer.Exit(1)
    n = registry.load_csv(facets_csv)
    console.print(f"[green]✓ Loaded {n} facets[/green]")

    # Check Ollama
    async with OllamaClient(base_url=ollama_url, model=model) as client:
        health = await client.health_check()
        if health["status"] != "ok":
            console.print(f"[red]Ollama not reachable at {ollama_url}[/red]")
            console.print(f"[yellow]Run: ollama serve  &&  ollama pull {model}[/yellow]")
            raise typer.Exit(1)

        if not health.get("model_available"):
            console.print(f"[yellow]⚠ Model '{model}' not found locally. Pulling...[/yellow]")

        console.print(f"[green]✓ Ollama OK — model: {model}[/green]")

        # Find conversation files
        conv_files = list(input_dir.glob("*.json"))
        if not conv_files:
            console.print(f"[red]No JSON files found in {input_dir}[/red]")
            raise typer.Exit(1)

        console.print(f"\n[bold]Scoring {len(conv_files)} conversations...[/bold]\n")

        orchestrator = ScoringOrchestrator(
            client=client, registry=registry, chunk_size=chunk_size
        )

        all_summaries = []
        for conv_file in track(conv_files, description="Scoring..."):
            try:
                data = load_conversation_file(str(conv_file))
                turns = split_conversation(
                    data["messages"],
                    conversation_id=data.get("id") or conv_file.stem,
                    score_user_turns=score_user_turns,
                )
                if not turns:
                    console.print(f"[yellow]No scoreable turns in {conv_file.name}[/yellow]")
                    continue

                results = await orchestrator.score_conversation(turns, facet_ids=facet_ids)
                paths = save_results(results, output_dir, conversation_id=conv_file.stem)
                summary = compute_summary(results)
                all_summaries.append({
                    "file": conv_file.name,
                    "turns": len(results),
                    "mean_score": summary["overall_mean_score"],
                    "mean_conf": summary["overall_mean_confidence"],
                    "safety_flags": len(summary["safety_flags"]),
                })
            except Exception as e:
                console.print(f"[red]Error on {conv_file.name}: {e}[/red]")

        # Summary table
        table = Table(title="Scoring Summary", show_header=True)
        table.add_column("File")
        table.add_column("Turns", justify="right")
        table.add_column("Mean Score", justify="right")
        table.add_column("Mean Conf", justify="right")
        table.add_column("Safety Flags", justify="right")

        for s in all_summaries:
            table.add_row(
                s["file"], str(s["turns"]),
                f"{s['mean_score']:.2f}", f"{s['mean_conf']:.2f}",
                str(s["safety_flags"]),
            )

        console.print()
        console.print(table)
        console.print(f"\n[green]✓ Results saved to {output_dir}[/green]")


if __name__ == "__main__":
    app()

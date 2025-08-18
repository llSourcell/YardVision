from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.traceback import install as install_rich_traceback

from vision.core.config import AppConfig, load_config
from vision.pipeline.processor import FrameProcessor


install_rich_traceback(show_locals=True)

app = typer.Typer(help="YardVision CLI")
console = Console()


@app.callback()
def main() -> None:
    """CLI entrypoint for YardVision."""


@app.command()
def version() -> None:
    """Print version and exit."""
    from . import __version__

    console.print(f"YardVision v{__version__}")


@app.command()
def run(
    source: str = typer.Option(
        "0", help="Video source: device index (e.g. '0') or path/URL"
    ),
    config: Optional[Path] = typer.Option(
        None, exists=True, file_okay=True, dir_okay=False, readable=True, help="Path to config file"
    ),
    display: bool = typer.Option(False, help="Display frames in a window (if available)"),
) -> None:
    """Run the YardVision pipeline."""
    try:
        app_config: AppConfig = load_config(config)
        processor = FrameProcessor(app_config=app_config)
        processor.process_stream(source=source, display=display)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()


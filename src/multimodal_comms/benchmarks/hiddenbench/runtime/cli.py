"""
Command-line interface for HiddenBench.
"""

import random
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console, Group
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.text import Text

from .benchmark import HiddenBench
from .config import Config
from .providers.factory import get_available_providers
from .report import generate_report, IncrementalReportWriter
from .task import Task, TaskInfo, load_tasks_from_directory, get_task_summary

app = typer.Typer(
    name="hiddenbench",
    help="HiddenBench: Evaluate collective reasoning in multi-agent LLM systems",
    add_completion=False,
    invoke_without_command=True,
)
console = Console()


def _format_tokens(count: int) -> str:
    """Format token count with K/M suffix for readability."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


@app.callback()
def main_callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit",
    ),
):
    """
    HiddenBench: Evaluate collective reasoning in multi-agent LLM systems.

    A benchmark based on the Hidden Profile paradigm from social psychology,
    where information is distributed across agents such that no single agent
    can determine the correct answer alone.
    """
    if version:
        console.print("hiddenbench version 0.1.0")
        raise typer.Exit()

    # If no command is given, show help
    if ctx.invoked_subcommand is None:
        console.print()
        console.print("[bold blue]HiddenBench[/bold blue] - Multi-Agent LLM Collective Reasoning Benchmark")
        console.print()
        console.print("Based on the Hidden Profile paradigm from social psychology research.")
        console.print()
        console.print("[bold]Commands:[/bold]")
        console.print("  [cyan]run[/cyan]             Run the benchmark evaluation")
        console.print("  [cyan]tasks[/cyan]           Show configured tasks and data sources")
        console.print("  [cyan]list-providers[/cyan]  List available LLM providers")
        console.print("  [cyan]create-task[/cyan]     Create a new task interactively")
        console.print("  [cyan]init[/cyan]            Initialize a new HiddenBench project")
        console.print()
        console.print("[bold]Quick Start:[/bold]")
        console.print("  1. Configure API keys in [cyan]config.yaml[/cyan]")
        console.print("  2. Run [cyan]hiddenbench tasks[/cyan] to verify data sources")
        console.print("  3. Run [cyan]hiddenbench run[/cyan] to start evaluation")
        console.print()
        console.print("[bold]Common Options (for 'run' command):[/bold]")
        console.print("  [cyan]-n, --num-tasks[/cyan]    Number of tasks to run (default: 1)")
        console.print("  [cyan]--all[/cyan]              Run all available tasks")
        console.print("  [cyan]--task[/cyan]             Run a specific task by name")
        console.print("  [cyan]-p, --provider[/cyan]     LLM provider (anthropic, openai, grok, local)")
        console.print("  [cyan]-m, --model[/cyan]        Model to use")
        console.print("  [cyan]-a, --agents[/cyan]       Number of agents (default: 4)")
        console.print("  [cyan]-r, --rounds[/cyan]       Discussion rounds (default: 15)")
        console.print("  [cyan]--no-early-stop[/cyan]    Run all rounds (don't stop on consensus)")
        console.print("  [cyan]--no-full-profile[/cyan]  Skip Full Profile baseline")
        console.print("  [cyan]--seed[/cyan]             Random seed for reproducibility")
        console.print("  [cyan]-v, --verbose[/cyan]      Show detailed output")
        console.print()
        console.print("Use [cyan]hiddenbench run --help[/cyan] for all options")
        console.print()


def _load_all_tasks(config: Config) -> list[Task]:
    """Load tasks from all configured sources."""
    tasks = []
    num_agents = config.benchmark.num_agents

    # Load official HiddenBench data
    if config.benchmark.use_official_data:
        data_dir = Path(config.benchmark.data_dir)
        if data_dir.exists():
            official_tasks = load_tasks_from_directory(data_dir, num_agents)
            tasks.extend(official_tasks)

    # Load custom tasks
    if config.benchmark.use_custom_tasks:
        tasks_dir = Path(config.benchmark.tasks_dir)
        if tasks_dir.exists():
            custom_tasks = load_tasks_from_directory(tasks_dir, num_agents)
            # Filter out duplicates by ID
            existing_ids = {t.id for t in tasks}
            for task in custom_tasks:
                if task.id not in existing_ids:
                    tasks.append(task)

    return tasks


@app.command()
def run(
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file (default: config.yaml)",
    ),
    data_dir: Optional[Path] = typer.Option(
        None,
        "--data",
        "-d",
        help="Directory containing official HiddenBench data",
    ),
    tasks_dir: Optional[Path] = typer.Option(
        None,
        "--tasks",
        "-t",
        help="Directory containing custom task files",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Directory for output reports",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="LLM provider to use (anthropic, openai, grok, local)",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use (overrides config)",
    ),
    num_agents: Optional[int] = typer.Option(
        None,
        "--agents",
        "-a",
        help="Number of agents (default: 4)",
    ),
    num_rounds: Optional[int] = typer.Option(
        None,
        "--rounds",
        "-r",
        help="Number of discussion rounds (default: 15)",
    ),
    no_full_profile: bool = typer.Option(
        False,
        "--no-full-profile",
        help="Skip Full Profile baseline evaluation",
    ),
    official_only: bool = typer.Option(
        False,
        "--official-only",
        help="Only use official HiddenBench tasks",
    ),
    custom_only: bool = typer.Option(
        False,
        "--custom-only",
        help="Only use custom tasks",
    ),
    num_tasks: Optional[int] = typer.Option(
        None,
        "--num-tasks",
        "-n",
        help="Number of tasks to run (default: 1, use 0 or 'all' for all tasks)",
    ),
    all_tasks: bool = typer.Option(
        False,
        "--all",
        help="Run all tasks (equivalent to --num-tasks 0)",
    ),
    task_name: Optional[str] = typer.Option(
        None,
        "--task",
        help="Run a specific task by name (case-insensitive, partial match supported)",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Random seed for task selection (for reproducibility)",
    ),
    no_early_stop: bool = typer.Option(
        False,
        "--no-early-stop",
        help="Don't stop early when consensus is reached (run all rounds like original paper)",
    ),
    protocol: Optional[str] = typer.Option(
        None,
        "--protocol",
        help="Channel protocol: full_history (original) or delta (no retransmission)",
    ),
    compressor: Optional[str] = typer.Option(
        None,
        "--compressor",
        help="Channel compressor: identity, window, novelty, llmlingua2, grammar, ...",
    ),
    message_style: Optional[str] = typer.Option(
        None,
        "--message-style",
        help="Sender message style: default, concise, schema",
    ),
    window_rounds: Optional[int] = typer.Option(
        None,
        "--window-rounds",
        help="Rounds kept by the window compressor",
    ),
    novelty_threshold: Optional[float] = typer.Option(
        None,
        "--novelty-threshold",
        help="Cosine similarity threshold for the novelty compressor",
    ),
    novelty_stateful: bool = typer.Option(
        False,
        "--novelty-stateful",
        help="Novelty compressor remembers per-receiver shown content across turns",
    ),
    lingua_rate: Optional[float] = typer.Option(
        None,
        "--lingua-rate",
        help="Compression rate for the llmlingua2 compressor (fraction of tokens kept)",
    ),
    selector_model: Optional[str] = typer.Option(
        None,
        "--selector-model",
        help="Checkpoint used by learned sentence-selection methods",
    ),
    selector_rate: Optional[float] = typer.Option(
        None,
        "--selector-rate",
        help="Char budget for the learned compressor (fraction of view kept)",
    ),
    selector_dedup: bool = typer.Option(
        False,
        "--selector-dedup",
        help="Learned compressor also drops content already shown to the receiver",
    ),
    rewriter_rate: Optional[float] = typer.Option(
        None,
        "--rewriter-rate",
        help="Char budget for the GRPO rewriter compressor (fraction of view kept)",
    ),
    backref_threshold: Optional[float] = typer.Option(
        None,
        "--backref-threshold",
        help="Cosine similarity threshold for the backref compressor",
    ),
    backref_drop_floor: Optional[float] = typer.Option(
        None,
        "--backref-drop-floor",
        help="Selector score below which backref drops a sentence outright (0 disables)",
    ),
    adaptive_drop_floor: Optional[float] = typer.Option(
        None,
        "--adaptive-drop-floor",
        help="Relevance score below which the adaptive compressor drops a sentence outright",
    ),
    adaptive_thr_lo: Optional[float] = typer.Option(
        None,
        "--adaptive-thr-lo",
        help="Dedup similarity threshold applied at relevance score 0 (adaptive compressor)",
    ),
    stack: Optional[str] = typer.Option(
        None,
        "--stack",
        help="Comma-separated compressor pipeline for --compressor stack (e.g. backref,codebook)",
    ),
    counterfactual_tau: Optional[float] = typer.Option(
        None,
        "--counterfactual-tau",
        help="Threshold on the distilled counterfactual-importance score (rate knob)",
    ),
    repmatch_tau: Optional[float] = typer.Option(
        None,
        "--repmatch-tau",
        help="Threshold on the distilled representational-match selector score (repmatch_selector)",
    ),
    repmatch_selector_model: Optional[str] = typer.Option(
        None,
        "--repmatch-selector-model",
        help="Checkpoint used by the representation-match selector",
    ),
    saliency_rate: Optional[float] = typer.Option(
        None,
        "--saliency-rate",
        help="Char-budget fraction for the saliency compressor",
    ),
    repserver_url: Optional[str] = typer.Option(
        None,
        "--repserver-url",
        help="Base URL of the representation server (saliency, repmatch_bestofk)",
    ),
    repmatch_rewriter_rate: Optional[float] = typer.Option(
        None,
        "--repmatch-rewriter-rate",
        help="Target char-budget fraction for the repmatch_rewriter compressor",
    ),
    tokenfilter_tau: Optional[float] = typer.Option(
        None,
        "--tokenfilter-tau",
        help="Probability threshold for the tokenfilter compressor (higher = more deletion)",
    ),
    tokenfilter_device: Optional[str] = typer.Option(
        None,
        "--tokenfilter-device",
        help="Device for the tokenfilter compressor (e.g., cuda:3)",
    ),
    autoencoder_model: Optional[str] = typer.Option(
        None,
        "--autoencoder-model",
        help="Checkpoint used by the sampled-latent autoencoder compressor",
    ),
    autoencoder_num_latents: Optional[int] = typer.Option(
        None,
        "--autoencoder-num-latents",
        help="Number of latent codebook tokens per message for autoencoder compressor",
    ),
    autoencoder_device: Optional[str] = typer.Option(
        None,
        "--autoencoder-device",
        help="Device for the autoencoder compressor (e.g., cuda:3)",
    ),
    mwnot_autoencoder_model: Optional[str] = typer.Option(
        None,
        "--mwnot-autoencoder-model",
        help="Checkpoint used by the MWNOT autoencoder compressor",
    ),
    mwnot_autoencoder_num_latents: Optional[int] = typer.Option(
        None,
        "--mwnot-autoencoder-num-latents",
        help="Number of generator vectors per MWNOT message",
    ),
    mwnot_autoencoder_device: Optional[str] = typer.Option(
        None,
        "--mwnot-autoencoder-device",
        help="Device for the MWNOT autoencoder compressor",
    ),
    grammar_codebook: Optional[str] = typer.Option(
        None,
        "--grammar-codebook",
        help="Path to the BPE grammar codebook JSON for the grammar compressor",
    ),
    certspan_eps: Optional[float] = typer.Option(
        None,
        "--certspan-eps",
        help="Certificate slack (1-cos budget) for the certspan compressor",
    ),
    semfallback_ladder: Optional[str] = typer.Option(
        None,
        "--semfallback-ladder",
        help="Compressor ladder for semfallback, most->least aggressive "
             "(e.g. 'window:window_rounds=1;novelty:threshold=0.75;identity')",
    ),
    semfallback_eps: Optional[float] = typer.Option(
        None,
        "--semfallback-eps",
        help="Certificate slack (1-cos budget) for the semfallback compressor",
    ),
    pdiff_model: Optional[str] = typer.Option(
        None,
        "--pdiff-model",
        help="Shared frozen LM for the pdiff/ratediff compressors",
    ),
    pdiff_device: Optional[str] = typer.Option(
        None,
        "--pdiff-device",
        help="Device for the pdiff/ratediff compressors (e.g., cuda:3)",
    ),
    telegraphic_eps: Optional[float] = typer.Option(
        None,
        "--telegraphic-eps",
        help="Certificate slack (1-cos budget) for the telegraphic compressor's reinflation",
    ),
    telegraphic_device: Optional[str] = typer.Option(
        None,
        "--telegraphic-device",
        help="Device for the telegraphic compressor's reinflation model (e.g., cuda:3)",
    ),
    ratediff_eps: Optional[float] = typer.Option(
        None,
        "--ratediff-eps",
        help="Certificate slack (1-cos budget) for omitting corrections in ratediff (0 = lossless)",
    ),
    superpose_model: Optional[str] = typer.Option(
        None,
        "--superpose-model",
        help="Checkpoint used by the learned superposition compressor",
    ),
    superpose_device: Optional[str] = typer.Option(
        None,
        "--superpose-device",
        help="Device for the learned superposition compressor",
    ),
    report_name: Optional[str] = typer.Option(
        None,
        "--report-name",
        help="Base name for the report files (default: timestamped)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
):
    """
    Run the HiddenBench evaluation.

    Evaluates collective reasoning in multi-agent LLM systems using the
    Hidden Profile paradigm.
    """
    try:
        # Load configuration
        config = Config.load(config_path)

        # Apply command-line overrides
        if provider:
            config.benchmark.provider = provider
        if model:
            config.benchmark.model = model
        if num_agents:
            config.benchmark.num_agents = num_agents
        if num_rounds:
            config.benchmark.num_rounds = num_rounds
        if no_full_profile:
            config.benchmark.run_full_profile = False
        if data_dir:
            config.benchmark.data_dir = str(data_dir)
        if tasks_dir:
            config.benchmark.tasks_dir = str(tasks_dir)
        if output_dir:
            config.benchmark.reports_dir = str(output_dir)
        if official_only:
            config.benchmark.use_official_data = True
            config.benchmark.use_custom_tasks = False
        if custom_only:
            config.benchmark.use_official_data = False
            config.benchmark.use_custom_tasks = True
        if protocol:
            config.channel.protocol = protocol
        if compressor:
            config.channel.compressor = compressor
        if message_style:
            config.channel.message_style = message_style
        if window_rounds is not None:
            config.channel.window_rounds = window_rounds
        if novelty_threshold is not None:
            config.channel.novelty_threshold = novelty_threshold
        if novelty_stateful:
            config.channel.novelty_stateful = True
        if lingua_rate is not None:
            config.channel.lingua_rate = lingua_rate
        if selector_model is not None:
            config.channel.selector_model = selector_model
        if selector_rate is not None:
            config.channel.selector_rate = selector_rate
        if selector_dedup:
            config.channel.selector_dedup = True
        if rewriter_rate is not None:
            config.channel.rewriter_rate = rewriter_rate
        if backref_threshold is not None:
            config.channel.backref_threshold = backref_threshold
        if backref_drop_floor is not None:
            config.channel.backref_drop_floor = backref_drop_floor
        if adaptive_drop_floor is not None:
            config.channel.adaptive_drop_floor = adaptive_drop_floor
        if adaptive_thr_lo is not None:
            config.channel.adaptive_thr_lo = adaptive_thr_lo
        if stack is not None:
            config.channel.stack = stack
        if counterfactual_tau is not None:
            config.channel.counterfactual_tau = counterfactual_tau
        if repmatch_tau is not None:
            config.channel.repmatch_tau = repmatch_tau
        if repmatch_selector_model is not None:
            config.channel.repmatch_selector_model = repmatch_selector_model
        if saliency_rate is not None:
            config.channel.saliency_rate = saliency_rate
        if repserver_url is not None:
            config.channel.repserver_url = repserver_url
        if repmatch_rewriter_rate is not None:
            config.channel.repmatch_rewriter_rate = repmatch_rewriter_rate
        if tokenfilter_tau is not None:
            config.channel.tokenfilter_tau = tokenfilter_tau
        if tokenfilter_device is not None:
            config.channel.tokenfilter_device = tokenfilter_device
        if autoencoder_model is not None:
            config.channel.autoencoder_model = autoencoder_model
        if autoencoder_num_latents is not None:
            config.channel.autoencoder_num_latents = autoencoder_num_latents
        if autoencoder_device is not None:
            config.channel.autoencoder_device = autoencoder_device
        if mwnot_autoencoder_model is not None:
            config.channel.mwnot_autoencoder_model = mwnot_autoencoder_model
        if mwnot_autoencoder_num_latents is not None:
            config.channel.mwnot_autoencoder_num_latents = mwnot_autoencoder_num_latents
        if mwnot_autoencoder_device is not None:
            config.channel.mwnot_autoencoder_device = mwnot_autoencoder_device
        if grammar_codebook is not None:
            config.channel.grammar_codebook = grammar_codebook
        if certspan_eps is not None:
            config.channel.certspan_eps = certspan_eps
        if semfallback_ladder is not None:
            config.channel.semfallback_ladder = semfallback_ladder
        if semfallback_eps is not None:
            config.channel.semfallback_eps = semfallback_eps
        if pdiff_model is not None:
            config.channel.pdiff_model = pdiff_model
        if pdiff_device is not None:
            config.channel.pdiff_device = pdiff_device
        if telegraphic_eps is not None:
            config.channel.telegraphic_eps = telegraphic_eps
        if telegraphic_device is not None:
            config.channel.telegraphic_device = telegraphic_device
        if ratediff_eps is not None:
            config.channel.ratediff_eps = ratediff_eps
        if superpose_model is not None:
            config.channel.superpose_model = superpose_model
        if superpose_device is not None:
            config.channel.superpose_device = superpose_device

        # Validate configuration
        config.validate()

        console.print()
        console.print("[bold blue]HiddenBench Evaluation[/bold blue]")
        console.print()

        # Display configuration
        console.print(f"[dim]Provider:[/dim] {config.benchmark.provider}")
        console.print(f"[dim]Model:[/dim] {config.benchmark.model or 'default'}")
        console.print(f"[dim]Agents:[/dim] {config.benchmark.num_agents}")
        console.print(f"[dim]Discussion Rounds:[/dim] {config.benchmark.num_rounds}")
        console.print(f"[dim]Full Profile:[/dim] {config.benchmark.run_full_profile}")
        channel_desc = config.channel.to_dict()
        if (config.channel.protocol != "full_history"
                or config.channel.compressor != "identity"
                or config.channel.message_style != "default"):
            console.print(f"[dim]Channel:[/dim] {channel_desc}")
        console.print()

        # Load tasks
        all_available_tasks = _load_all_tasks(config)

        if not all_available_tasks:
            console.print("[red]Error:[/red] No tasks found")
            console.print()
            console.print("Task sources checked:")
            if config.benchmark.use_official_data:
                console.print(f"  - Official data: {config.benchmark.data_dir}")
            if config.benchmark.use_custom_tasks:
                console.print(f"  - Custom tasks: {config.benchmark.tasks_dir}")
            console.print()
            console.print("Run [cyan]hiddenbench tasks[/cyan] to see available tasks")
            raise typer.Exit(1)

        # Determine which tasks to run
        total_available = len(all_available_tasks)

        # Check if a specific task was requested
        if task_name:
            # Search for task by name (case-insensitive, partial match)
            search_term = task_name.lower()
            matching_tasks = [
                t for t in all_available_tasks
                if search_term in t.name.lower()
            ]

            if not matching_tasks:
                console.print(f"[red]Error:[/red] No task found matching '{task_name}'")
                console.print()
                console.print("[dim]Available tasks:[/dim]")
                for t in all_available_tasks[:10]:
                    console.print(f"  - {t.name}")
                if len(all_available_tasks) > 10:
                    console.print(f"  ... and {len(all_available_tasks) - 10} more")
                console.print()
                console.print("Run [cyan]hiddenbench tasks --verbose[/cyan] to see all tasks")
                raise typer.Exit(1)
            elif len(matching_tasks) > 1:
                console.print(f"[yellow]Multiple tasks match '{task_name}':[/yellow]")
                for t in matching_tasks:
                    console.print(f"  - {t.name}")
                console.print()
                console.print("[dim]Please provide a more specific name[/dim]")
                raise typer.Exit(1)
            else:
                tasks = matching_tasks
                console.print(f"[green]Running specific task:[/green] {tasks[0].name}")
        elif all_tasks or num_tasks == 0:
            # Run all tasks
            tasks = all_available_tasks
            console.print(f"[green]Running all {total_available} tasks[/green]")
        elif num_tasks is None:
            # Default: run 1 random task
            tasks_to_run = 1
            if seed is not None:
                random.seed(seed)
            tasks = random.sample(all_available_tasks, tasks_to_run)
            console.print(f"[green]Selected {tasks_to_run} random task(s) from {total_available} available[/green]")
            if seed is not None:
                console.print(f"  [dim]Random seed: {seed}[/dim]")
        else:
            # Run specified number of tasks
            tasks_to_run = min(num_tasks, total_available)
            if seed is not None:
                random.seed(seed)
            tasks = random.sample(all_available_tasks, tasks_to_run)
            console.print(f"[green]Selected {tasks_to_run} random task(s) from {total_available} available[/green]")
            if seed is not None:
                console.print(f"  [dim]Random seed: {seed}[/dim]")

        # Show task summary
        summary = get_task_summary(all_available_tasks)
        for source, count in summary["sources"].items():
            console.print(f"  [dim]{source}:[/dim] {count}")
        console.print()

        # Initialize benchmark
        bench = HiddenBench(config)

        # Calculate total LLM calls for progress estimation
        num_agents_cfg = config.benchmark.num_agents
        num_rounds_cfg = config.benchmark.num_rounds
        run_fp = config.benchmark.run_full_profile
        calls_per_task = (
            num_agents_cfg +                           # pre-discussion
            num_rounds_cfg * num_agents_cfg +          # discussion
            num_agents_cfg +                           # post-discussion
            (num_agents_cfg if run_fp else 0)          # full profile
        )
        total_calls = len(tasks) * calls_per_task

        # Estimate tokens using provider-specific parameters
        # The key insight: discussion context grows with each message
        est_params = config.get_token_estimation_params()
        base_input = est_params["est_base_input_tokens"]
        output_per_call = est_params["est_output_tokens"]
        context_growth = est_params["est_context_growth"]

        def estimate_task_tokens() -> tuple[int, int]:
            """Estimate input and output tokens for one task."""
            est_input = 0
            est_output = 0

            # Pre-discussion: each agent gets base input, produces output
            est_input += num_agents_cfg * base_input
            est_output += num_agents_cfg * output_per_call

            # Discussion: context grows as messages accumulate
            # Each message adds ~context_growth tokens to subsequent calls
            discussion_calls = num_rounds_cfg * num_agents_cfg
            for msg_idx in range(discussion_calls):
                # Base input + accumulated context from prior messages
                accumulated_context = msg_idx * context_growth
                est_input += base_input + accumulated_context
                est_output += output_per_call

            # Post-discussion: full context + base input
            full_discussion_context = discussion_calls * context_growth
            est_input += num_agents_cfg * (base_input + full_discussion_context)
            est_output += num_agents_cfg * output_per_call

            # Full profile (if enabled): just base input, no discussion context
            if run_fp:
                est_input += num_agents_cfg * base_input
                est_output += num_agents_cfg * output_per_call

            return est_input, est_output

        est_input_per_task, est_output_per_task = estimate_task_tokens()
        est_total_input = len(tasks) * est_input_per_task
        est_total_output = len(tasks) * est_output_per_task
        est_total_tokens = est_total_input + est_total_output

        console.print(f"[dim]Estimated LLM calls: {total_calls} ({calls_per_task} per task)[/dim]")
        if no_early_stop:
            console.print(f"[dim]Estimated tokens: ~{_format_tokens(est_total_tokens)} (in: ~{_format_tokens(est_total_input)}, out: ~{_format_tokens(est_total_output)})[/dim]")
        else:
            console.print(f"[dim]Estimated max tokens: ~{_format_tokens(est_total_tokens)} (in: ~{_format_tokens(est_total_input)}, out: ~{_format_tokens(est_total_output)}) if all rounds run[/dim]")
        console.print()

        # Run evaluation with rich progress display
        results = []

        # Create progress display
        overall_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        )

        phase_progress = Progress(
            SpinnerColumn(),
            TextColumn("  {task.description}"),
            BarColumn(bar_width=20),
            TextColumn("[dim]{task.fields[detail]}[/dim]"),
            console=console,
        )

        token_progress = Progress(
            TextColumn("  [cyan]Tokens:[/cyan] {task.fields[tokens]}"),
            console=console,
        )

        status_progress = Progress(
            TextColumn("  {task.fields[status]}"),
            console=console,
        )

        # Phase descriptions for display
        phase_names = {
            "init": "Initializing",
            "pre_discussion": "Pre-discussion decisions",
            "discussion": "Discussion phase",
            "post_discussion": "Post-discussion decisions",
            "full_profile": "Full Profile baseline",
        }

        # Initialize incremental report writer
        report_config = {
            "provider": config.benchmark.provider,
            "model": config.benchmark.model or "default",
            "num_agents": config.benchmark.num_agents,
            "num_rounds": config.benchmark.num_rounds,
            "temperature": config.benchmark.temperature,
            "run_full_profile": config.benchmark.run_full_profile,
            "data_sources": {
                "official_data": config.benchmark.use_official_data,
                "custom_tasks": config.benchmark.use_custom_tasks,
            },
            "channel": config.channel.to_dict(),
            "tasks_selected": len(tasks),
            "tasks_available": total_available,
        }
        report_writer = IncrementalReportWriter(
            config=report_config,
            output_dir=config.benchmark.reports_dir,
            report_name=report_name,
        )

        with Live(Group(overall_progress, phase_progress, token_progress, status_progress), console=console, refresh_per_second=4):
            overall_task = overall_progress.add_task(
                f"Running benchmark (0/{len(tasks)} tasks)",
                total=len(tasks),
            )

            phase_task = phase_progress.add_task(
                "Preparing...",
                total=100,
                detail="",
            )

            token_task = token_progress.add_task(
                "Tokens",
                total=100,
                tokens="0 (input: 0, output: 0)",
            )

            status_task = status_progress.add_task(
                "Status",
                total=100,
                status="",
            )

            # Create status callback for rate limiting messages
            def status_callback(status_type: str, message: str):
                if status_type == "rate_limit":
                    status_progress.update(status_task, status=f"[yellow]⏳ {message}[/yellow]")
                elif status_type == "retry":
                    status_progress.update(status_task, status=f"[yellow]🔄 {message}[/yellow]")
                else:
                    status_progress.update(status_task, status=f"[dim]{message}[/dim]")

            # Set up status callback on benchmark
            bench.set_status_callback(status_callback)

            for task_idx, task in enumerate(tasks):
                # Clear status message for new task
                status_progress.update(status_task, status="")

                # Update overall progress
                overall_progress.update(
                    overall_task,
                    description=f"Task {task_idx + 1}/{len(tasks)}: {task.name[:40]}{'...' if len(task.name) > 40 else ''}",
                )

                # Create progress callback
                def progress_callback(phase: str, current: int, total: int, detail: str, tokens: dict[str, int]):
                    phase_name = phase_names.get(phase, phase)
                    pct = (current / total * 100) if total > 0 else 0
                    phase_progress.update(
                        phase_task,
                        description=f"{phase_name}",
                        completed=pct,
                        detail=detail,
                    )
                    # Update token display
                    token_progress.update(
                        token_task,
                        tokens=f"{_format_tokens(tokens['total'])} (in: {_format_tokens(tokens['input'])}, out: {_format_tokens(tokens['output'])})",
                    )

                try:
                    result = bench.evaluate_task(
                        task,
                        progress_callback=progress_callback,
                        early_stop_on_consensus=not no_early_stop,
                    )
                    results.append(result)

                    # Save result incrementally
                    report_writer.add_result(result)

                    # Show task result
                    overall_progress.advance(overall_task)

                    # Build consensus info string
                    consensus_info = ""
                    if result.consensus_round is not None:
                        consensus_info = f" Consensus=R{result.consensus_round + 1}"
                        if result.stopped_early:
                            consensus_info += " (stopped)"
                    else:
                        consensus_info = f" Rounds={result.total_rounds}"

                    if verbose:
                        # Print detailed results for this task
                        console.print(
                            f"  [green]✓[/green] {task.name}: "
                            f"Pre={result.pre_accuracy:.0%} "
                            f"Post={result.post_accuracy:.0%} "
                            f"Gain={result.information_integration_gain:+.0%}"
                            + (f" FP={result.full_profile_accuracy:.0%}" if result.full_profile_accuracy is not None else "")
                            + consensus_info
                        )
                    else:
                        # Brief status indicator
                        gain = result.information_integration_gain
                        if gain > 0:
                            status = "[green]↑[/green]"
                        elif gain < 0:
                            status = "[red]↓[/red]"
                        else:
                            status = "[yellow]→[/yellow]"
                        console.print(f"  {status} Task {task_idx + 1}: {result.post_accuracy:.0%} accuracy{consensus_info}")

                except Exception as e:
                    console.print(f"  [red]✗[/red] Error on task {task.id}: {e}")
                    if verbose:
                        import traceback
                        traceback.print_exc()
                    overall_progress.advance(overall_task)

            # Mark completion
            phase_progress.update(phase_task, description="Complete", completed=100, detail="")
            status_progress.update(status_task, status="")
            final_tokens = bench.get_token_usage()

        if not results:
            console.print("[red]No tasks completed successfully[/red]")
            console.print(f"[dim]Partial report saved to: {report_writer.json_path}[/dim]")
            raise typer.Exit(1)

        # Finalize the incremental report
        json_path, md_path = report_writer.finalize(token_usage=final_tokens)

        # Display summary
        console.print()
        console.print("[bold green]Evaluation Complete![/bold green]")
        console.print()

        # Summary table
        table = Table(title="Results Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        avg_pre = sum(r.pre_accuracy for r in results) / len(results)
        avg_post = sum(r.post_accuracy for r in results) / len(results)
        avg_gain = sum(r.information_integration_gain for r in results) / len(results)

        table.add_row("Tasks Evaluated", str(len(results)))
        table.add_row("Pre-Discussion Accuracy", f"{avg_pre:.1%}")
        table.add_row("Post-Discussion Accuracy", f"{avg_post:.1%}")
        table.add_row("Information Gain", f"{avg_gain:+.1%}")

        fp_results = [r for r in results if r.full_profile_accuracy is not None]
        if fp_results:
            avg_fp = sum(r.full_profile_accuracy for r in fp_results) / len(fp_results)
            table.add_row("Full Profile Accuracy", f"{avg_fp:.1%}")

        # Consensus statistics
        consensus_results = [r for r in results if r.consensus_round is not None]
        if consensus_results:
            avg_consensus_round = sum(r.consensus_round + 1 for r in consensus_results) / len(consensus_results)
            table.add_row("", "")  # Spacer row
            table.add_row("Tasks with Consensus", f"{len(consensus_results)}/{len(results)}")
            table.add_row("Avg Consensus Round", f"{avg_consensus_round:.1f}")
            early_stopped = sum(1 for r in results if r.stopped_early)
            if early_stopped > 0:
                table.add_row("Tasks Stopped Early", str(early_stopped))

        # Token usage
        table.add_row("", "")  # Spacer row
        table.add_row("Total Tokens", f"{final_tokens['total']:,}")
        table.add_row("  Input Tokens", f"{final_tokens['input']:,}")
        table.add_row("  Output Tokens", f"{final_tokens['output']:,}")

        console.print(table)
        console.print()

        console.print(f"[dim]JSON Report:[/dim] {json_path}")
        console.print(f"[dim]Markdown Report:[/dim] {md_path}")
        console.print()

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print()
        console.print("Create a configuration file by copying the example:")
        console.print("  [cyan]cp config.example.yaml config.yaml[/cyan]")
        console.print("Then edit config.yaml to add your API keys.")
        raise typer.Exit(1)

    except ValueError as e:
        console.print(f"[red]Configuration Error:[/red] {e}")
        raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def list_providers():
    """List available LLM providers."""
    console.print()
    console.print("[bold]Available LLM Providers:[/bold]")
    console.print()

    providers = get_available_providers()
    for provider in providers:
        console.print(f"  - {provider}")

    console.print()
    console.print("[dim]Configure providers in config.yaml[/dim]")
    console.print()


@app.command("tasks")
def show_tasks(
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    data_dir: Optional[Path] = typer.Option(
        None,
        "--data",
        "-d",
        help="Directory containing official HiddenBench data",
    ),
    tasks_dir: Optional[Path] = typer.Option(
        None,
        "--tasks",
        "-t",
        help="Directory containing custom task files",
    ),
    official_only: bool = typer.Option(
        False,
        "--official-only",
        help="Only show official HiddenBench tasks",
    ),
    custom_only: bool = typer.Option(
        False,
        "--custom-only",
        help="Only show custom tasks",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed task information",
    ),
):
    """
    Show configured tasks and data sources.

    Lists all tasks that will be used when running the benchmark,
    along with their sources and basic information.
    """
    # Load configuration
    try:
        config = Config.load(config_path)
    except FileNotFoundError:
        # Use defaults if no config file
        config = Config()

    # Apply command-line overrides
    if data_dir:
        config.benchmark.data_dir = str(data_dir)
    if tasks_dir:
        config.benchmark.tasks_dir = str(tasks_dir)
    if official_only:
        config.benchmark.use_official_data = True
        config.benchmark.use_custom_tasks = False
    if custom_only:
        config.benchmark.use_official_data = False
        config.benchmark.use_custom_tasks = True

    console.print()
    console.print("[bold blue]HiddenBench Task Configuration[/bold blue]")
    console.print()

    # Show data source configuration
    console.print("[bold]Data Sources:[/bold]")
    if config.benchmark.use_official_data:
        data_path = Path(config.benchmark.data_dir)
        exists = data_path.exists()
        status = "[green]OK[/green]" if exists else "[red]NOT FOUND[/red]"
        console.print(f"  Official data: {data_path} {status}")
        if exists:
            benchmark_file = data_path / "benchmark.json"
            if benchmark_file.exists():
                console.print(f"    [dim]benchmark.json found[/dim]")
    else:
        console.print("  Official data: [dim]disabled[/dim]")

    if config.benchmark.use_custom_tasks:
        tasks_path = Path(config.benchmark.tasks_dir)
        exists = tasks_path.exists()
        status = "[green]OK[/green]" if exists else "[yellow]EMPTY[/yellow]"
        console.print(f"  Custom tasks: {tasks_path} {status}")
    else:
        console.print("  Custom tasks: [dim]disabled[/dim]")

    console.print()

    # Load tasks
    tasks = _load_all_tasks(config)

    if not tasks:
        console.print("[yellow]No tasks found in configured sources[/yellow]")
        console.print()
        console.print("To get started:")
        console.print("  1. Ensure the official HiddenBench data is in:")
        console.print(f"     [cyan]{config.benchmark.data_dir}[/cyan]")
        console.print("  2. Or add custom tasks to:")
        console.print(f"     [cyan]{config.benchmark.tasks_dir}[/cyan]")
        return

    # Generate and display summary
    summary = get_task_summary(tasks)

    console.print(f"[bold]Total Tasks: {summary['total_tasks']}[/bold]")
    console.print()

    # Tasks by source
    if summary["sources"]:
        console.print("[bold]By Source:[/bold]")
        for source, count in sorted(summary["sources"].items()):
            console.print(f"  {source}: {count}")
        console.print()

    # Tasks by option count
    if summary["option_counts"]:
        console.print("[bold]By Answer Options:[/bold]")
        for n_options, count in sorted(summary["option_counts"].items()):
            console.print(f"  {n_options} options: {count} tasks")
        console.print()

    # Task list
    if verbose:
        console.print("[bold]Task Details:[/bold]")
        console.print()

        table = Table(show_header=True, header_style="bold")
        table.add_column("ID", style="cyan", width=12)
        table.add_column("Name", style="green", width=30)
        table.add_column("Options", width=25)
        table.add_column("Correct", style="yellow", width=15)
        table.add_column("Source", style="dim", width=15)

        for task in tasks:
            source = task.metadata.get("source", "custom")
            table.add_row(
                str(task.id),
                task.name[:28] + ".." if len(task.name) > 30 else task.name,
                ", ".join(task.options)[:23] + ".." if len(", ".join(task.options)) > 25 else ", ".join(task.options),
                task.correct_answer[:13] + ".." if len(task.correct_answer) > 15 else task.correct_answer,
                source,
            )

        console.print(table)
    else:
        # Brief list
        console.print("[bold]Tasks:[/bold]")
        for task in tasks[:10]:
            source = task.metadata.get("source", "custom")
            console.print(f"  [{source}] {task.id}: {task.name}")

        if len(tasks) > 10:
            console.print(f"  [dim]... and {len(tasks) - 10} more[/dim]")

        console.print()
        console.print("[dim]Use --verbose for detailed task list[/dim]")

    console.print()


@app.command()
def create_task(
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output path for the task JSON file",
    ),
):
    """Create a new task interactively."""
    console.print()
    console.print("[bold]Create a New HiddenBench Task[/bold]")
    console.print()

    task_id = typer.prompt("Task ID (unique identifier)")
    name = typer.prompt("Task name")
    description = typer.prompt("Scenario description")

    # Get options
    options = []
    console.print("Enter answer options (empty line to finish):")
    while True:
        option = typer.prompt(f"  Option {len(options) + 1}", default="")
        if not option:
            if len(options) < 2:
                console.print("[red]At least 2 options required[/red]")
                continue
            break
        options.append(option)

    correct_answer = typer.prompt(
        "Correct answer",
        type=str,
    )
    if correct_answer not in options:
        console.print(f"[yellow]Warning: '{correct_answer}' not in options list[/yellow]")

    num_agents = typer.prompt("Number of agents", default=3, type=int)

    # Get shared info
    shared_info = []
    console.print("Enter shared information (available to all agents, empty line to finish):")
    while True:
        info = typer.prompt(f"  Shared fact {len(shared_info) + 1}", default="")
        if not info:
            break
        shared_info.append({"content": info, "is_shared": True})

    # Get unshared info for each agent
    unshared_info = []
    for agent_idx in range(num_agents):
        agent_info = []
        console.print(f"Enter unique information for Agent {agent_idx + 1} (empty line to finish):")
        while True:
            info = typer.prompt(f"  Agent {agent_idx + 1} fact {len(agent_info) + 1}", default="")
            if not info:
                break
            agent_info.append({"content": info, "is_shared": False})
        unshared_info.append(agent_info)

    # Create task
    task = Task(
        id=task_id,
        name=name,
        description=description,
        options=options,
        correct_answer=correct_answer,
        shared_info=[
            TaskInfo(content=i["content"], is_shared=True)
            for i in shared_info
        ],
        unshared_info=[
            [
                TaskInfo(content=i["content"], is_shared=False)
                for i in agent_info
            ]
            for agent_info in unshared_info
        ],
    )

    task.save(output)
    console.print()
    console.print(f"[green]Task saved to {output}[/green]")


@app.command()
def init():
    """Initialize a new HiddenBench project."""
    console.print()
    console.print("[bold]Initializing HiddenBench Project[/bold]")
    console.print()

    # Create directories
    dirs = ["tasks", "reports", "data/hiddenbench_official"]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        console.print(f"  Created [cyan]{d}/[/cyan]")

    # Copy example config if it doesn't exist
    if not Path("config.yaml").exists():
        example_path = Path(__file__).parent.parent.parent.parent / "config.example.yaml"
        if example_path.exists():
            import shutil
            shutil.copy(example_path, "config.yaml")
            console.print("  Created [cyan]config.yaml[/cyan] from example")
        else:
            console.print("  [yellow]config.yaml not created - copy config.example.yaml manually[/yellow]")

    console.print()
    console.print("[green]Project initialized![/green]")
    console.print()
    console.print("Next steps:")
    console.print("  1. Download official HiddenBench data:")
    console.print("     [cyan]curl -L -o data/hiddenbench_official/benchmark.json \\")
    console.print("       https://huggingface.co/datasets/YuxuanLi1225/HiddenBench/resolve/main/benchmark.json[/cyan]")
    console.print("  2. Edit [cyan]config.yaml[/cyan] and add your API keys")
    console.print("  3. Run [cyan]hiddenbench tasks[/cyan] to verify task loading")
    console.print("  4. Run [cyan]hiddenbench run[/cyan] to start evaluation")
    console.print()


# Keep old list-tasks as alias for backwards compatibility
@app.command("list-tasks", hidden=True)
def list_tasks_legacy(
    tasks_dir: Path = typer.Option(
        Path("./tasks"),
        "--tasks",
        "-t",
        help="Directory containing task files",
    ),
):
    """List available tasks (deprecated, use 'tasks' command)."""
    tasks = load_tasks_from_directory(tasks_dir)

    if not tasks:
        console.print(f"[yellow]No tasks found in {tasks_dir}[/yellow]")
        return

    console.print()
    console.print(f"[bold]Tasks in {tasks_dir}:[/bold]")
    console.print()

    table = Table()
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Options", style="dim")
    table.add_column("Correct", style="yellow")

    for task in tasks:
        table.add_row(
            task.id,
            task.name,
            ", ".join(task.options),
            task.correct_answer,
        )

    console.print(table)
    console.print()


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()

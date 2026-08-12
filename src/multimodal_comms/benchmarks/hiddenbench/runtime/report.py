"""
Report generation for HiddenBench.

Generates comprehensive reports in JSON and Markdown formats.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .task import TaskResult


@dataclass
class BenchmarkReport:
    """
    Complete benchmark report.

    Contains all evaluation results, metrics, and full conversation logs.
    """
    results: list[TaskResult]
    config: dict[str, Any]
    metadata: dict[str, Any]

    @property
    def num_tasks(self) -> int:
        """Number of tasks evaluated."""
        return len(self.results)

    @property
    def average_pre_accuracy(self) -> float:
        """Average pre-discussion accuracy across all tasks."""
        if not self.results:
            return 0.0
        return sum(r.pre_accuracy for r in self.results) / len(self.results)

    @property
    def average_post_accuracy(self) -> float:
        """Average post-discussion accuracy across all tasks."""
        if not self.results:
            return 0.0
        return sum(r.post_accuracy for r in self.results) / len(self.results)

    @property
    def average_full_profile_accuracy(self) -> float | None:
        """Average Full Profile accuracy across all tasks."""
        fp_results = [r for r in self.results if r.full_profile_accuracy is not None]
        if not fp_results:
            return None
        return sum(r.full_profile_accuracy for r in fp_results) / len(fp_results)

    @property
    def pre_majority_rate(self) -> float:
        """Rate of correct majority decisions pre-discussion."""
        if not self.results:
            return 0.0
        correct = sum(1 for r in self.results if r.pre_majority_correct)
        return correct / len(self.results)

    @property
    def post_majority_rate(self) -> float:
        """Rate of correct majority decisions post-discussion."""
        if not self.results:
            return 0.0
        correct = sum(1 for r in self.results if r.post_majority_correct)
        return correct / len(self.results)

    @property
    def average_information_gain(self) -> float:
        """Average improvement from discussion."""
        if not self.results:
            return 0.0
        return sum(r.information_integration_gain for r in self.results) / len(self.results)

    @property
    def average_collective_gap(self) -> float | None:
        """Average gap between full profile and post-discussion."""
        gaps = [r.collective_reasoning_gap for r in self.results if r.collective_reasoning_gap is not None]
        if not gaps:
            return None
        return sum(gaps) / len(gaps)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "metadata": self.metadata,
            "config": self.config,
            "summary": {
                "num_tasks": self.num_tasks,
                "average_pre_accuracy": self.average_pre_accuracy,
                "average_post_accuracy": self.average_post_accuracy,
                "average_full_profile_accuracy": self.average_full_profile_accuracy,
                "pre_majority_rate": self.pre_majority_rate,
                "post_majority_rate": self.post_majority_rate,
                "average_information_gain": self.average_information_gain,
                "average_collective_gap": self.average_collective_gap,
            },
            "results": [r.to_dict() for r in self.results],
        }

    def save_json(self, path: str | Path) -> None:
        """Save report as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def save_markdown(self, path: str | Path) -> None:
        """Save report as Markdown."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        md = self._generate_markdown()
        with open(path, "w") as f:
            f.write(md)

    def _generate_markdown(self) -> str:
        """Generate Markdown report content."""
        lines = []

        # Header
        lines.append("# HiddenBench Evaluation Report")
        lines.append("")
        lines.append(f"**Generated:** {self.metadata.get('timestamp', 'Unknown')}")
        lines.append(f"**Provider:** {self.config.get('provider', 'Unknown')}")
        lines.append(f"**Model:** {self.config.get('model', 'Default')}")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Tasks Evaluated | {self.num_tasks} |")
        lines.append(f"| Pre-Discussion Accuracy (Avg) | {self.average_pre_accuracy:.2%} |")
        lines.append(f"| Post-Discussion Accuracy (Avg) | {self.average_post_accuracy:.2%} |")
        if self.average_full_profile_accuracy is not None:
            lines.append(f"| Full Profile Accuracy (Avg) | {self.average_full_profile_accuracy:.2%} |")
        lines.append(f"| Pre-Discussion Majority Correct | {self.pre_majority_rate:.2%} |")
        lines.append(f"| Post-Discussion Majority Correct | {self.post_majority_rate:.2%} |")
        lines.append(f"| Information Integration Gain | {self.average_information_gain:+.2%} |")
        if self.average_collective_gap is not None:
            lines.append(f"| Collective Reasoning Gap | {self.average_collective_gap:+.2%} |")
        lines.append("")

        # Results by task
        lines.append("## Results by Task")
        lines.append("")

        for i, result in enumerate(self.results, 1):
            task = result.task
            lines.append(f"### Task {i}: {task.name}")
            lines.append("")
            lines.append(f"**ID:** `{task.id}`")
            lines.append("")
            lines.append("**Scenario:**")
            lines.append(f"> {task.description}")
            lines.append("")
            lines.append(f"**Correct Answer:** {task.correct_answer}")
            lines.append("")

            # Metrics table
            lines.append("#### Metrics")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Pre-Discussion Accuracy | {result.pre_accuracy:.2%} |")
            lines.append(f"| Post-Discussion Accuracy | {result.post_accuracy:.2%} |")
            if result.full_profile_accuracy is not None:
                lines.append(f"| Full Profile Accuracy | {result.full_profile_accuracy:.2%} |")
            lines.append(f"| Information Gain | {result.information_integration_gain:+.2%} |")
            lines.append(f"| Discussion Rounds | {result.total_rounds} |")
            if result.consensus_round is not None:
                lines.append(f"| Consensus Reached | Round {result.consensus_round + 1} |")
            else:
                lines.append(f"| Consensus Reached | No |")
            if result.stopped_early:
                lines.append(f"| Early Stop | Yes |")
            lines.append("")

            # Initial Prompts (show only for first agent in detail, summarize others)
            if result.agent_initial_prompts:
                lines.append("#### Initial Prompts")
                lines.append("")
                lines.append("*The complete prompt given to Agent 1 at the start of the task:*")
                lines.append("")
                lines.append("```")
                lines.append(result.agent_initial_prompts[0])
                lines.append("```")
                lines.append("")
                if len(result.agent_initial_prompts) > 1:
                    lines.append(f"*Agents 2-{len(result.agent_initial_prompts)} received the same system prompt and scenario, but with different information sets (their unique hidden facts).*")
                    lines.append("")

            # Pre-discussion decisions
            lines.append("#### Pre-Discussion Decisions")
            lines.append("")
            for decision in result.pre_discussion_decisions:
                status = "Correct" if decision.is_correct else "Incorrect"
                lines.append(f"**Agent {decision.agent_id + 1}:** {decision.vote} ({status})")
                lines.append(f"> {decision.rationale}")
                lines.append("")

            # Discussion transcript
            lines.append("#### Discussion Transcript")
            lines.append("")
            if result.consensus_round is not None:
                consensus_info = f" (consensus reached in round {result.consensus_round + 1})"
                if result.stopped_early:
                    consensus_info += " - stopped early"
                lines.append(f"*{result.total_rounds} rounds completed{consensus_info}*")
                lines.append("")

            current_round = -1
            for msg in result.discussion_history:
                if msg.round_num != current_round:
                    current_round = msg.round_num
                    round_label = f"**--- Round {current_round + 1} ---**"
                    if result.consensus_round is not None and current_round == result.consensus_round:
                        round_label += " *(consensus reached)*"
                    lines.append(round_label)
                    lines.append("")

                lines.append(f"**Agent {msg.agent_id + 1}:**")
                lines.append(f"> {msg.content}")
                lines.append("")

            # Post-discussion decisions
            lines.append("#### Post-Discussion Decisions")
            lines.append("")
            for decision in result.post_discussion_decisions:
                status = "Correct" if decision.is_correct else "Incorrect"
                lines.append(f"**Agent {decision.agent_id + 1}:** {decision.vote} ({status})")
                lines.append(f"> {decision.rationale}")
                lines.append("")

            # Full Profile decisions (if available)
            if result.full_profile_decisions:
                lines.append("#### Full Profile Baseline Decisions")
                lines.append("")
                lines.append("*Each agent received all information and decided independently.*")
                lines.append("")
                for decision in result.full_profile_decisions:
                    status = "Correct" if decision.is_correct else "Incorrect"
                    lines.append(f"**Agent {decision.agent_id + 1}:** {decision.vote} ({status})")
                    lines.append(f"> {decision.rationale}")
                    lines.append("")

            lines.append("---")
            lines.append("")

        # Configuration
        lines.append("## Configuration")
        lines.append("")
        lines.append("```yaml")
        for key, value in self.config.items():
            lines.append(f"{key}: {value}")
        lines.append("```")
        lines.append("")

        return "\n".join(lines)


class IncrementalReportWriter:
    """
    Writes reports incrementally as tasks complete.

    This allows partial results to be saved even if the benchmark
    is interrupted or fails partway through.
    """

    def __init__(
        self,
        config: dict[str, Any],
        output_dir: str | Path,
        report_name: str | None = None,
    ):
        """
        Initialize the incremental report writer.

        Args:
            config: Configuration used for the benchmark.
            output_dir: Directory to save reports.
            report_name: Base name for report files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.timestamp = datetime.now()
        if report_name is None:
            report_name = f"hiddenbench_{self.timestamp.strftime('%Y%m%d_%H%M%S')}"

        self.report_name = report_name
        self.config = config
        self.metadata = {
            "timestamp": self.timestamp.isoformat(),
            "version": "0.1.0",
            "status": "in_progress",
        }

        self.results: list[TaskResult] = []
        self.json_path = self.output_dir / f"{report_name}.json"
        self.md_path = self.output_dir / f"{report_name}.md"

        # Write initial empty report
        self._write_reports()

    def add_result(self, result: TaskResult) -> None:
        """
        Add a task result and update the report files.

        Args:
            result: The completed task result.
        """
        self.results.append(result)
        self._write_reports()

    def finalize(self, token_usage: dict[str, int] | None = None) -> tuple[Path, Path]:
        """
        Mark the report as complete and write final version.

        Args:
            token_usage: Optional token usage statistics to include.

        Returns:
            Tuple of (json_path, markdown_path).
        """
        self.metadata["status"] = "complete"
        self.metadata["completed_at"] = datetime.now().isoformat()
        if token_usage:
            self.config["token_usage"] = token_usage
        self._write_reports()
        return self.json_path, self.md_path

    def _write_reports(self) -> None:
        """Write both JSON and Markdown reports with current results."""
        report = BenchmarkReport(
            results=self.results,
            config=self.config,
            metadata=self.metadata,
        )
        report.save_json(self.json_path)
        report.save_markdown(self.md_path)


def generate_report(
    results: list[TaskResult],
    config: dict[str, Any],
    output_dir: str | Path,
    report_name: str | None = None,
) -> tuple[Path, Path]:
    """
    Generate and save a benchmark report.

    Args:
        results: List of task results.
        config: Configuration used for the benchmark.
        output_dir: Directory to save reports.
        report_name: Base name for report files.

    Returns:
        Tuple of (json_path, markdown_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now()
    if report_name is None:
        report_name = f"hiddenbench_{timestamp.strftime('%Y%m%d_%H%M%S')}"

    metadata = {
        "timestamp": timestamp.isoformat(),
        "version": "0.1.0",
        "status": "complete",
    }

    report = BenchmarkReport(
        results=results,
        config=config,
        metadata=metadata,
    )

    json_path = output_dir / f"{report_name}.json"
    md_path = output_dir / f"{report_name}.md"

    report.save_json(json_path)
    report.save_markdown(md_path)

    return json_path, md_path

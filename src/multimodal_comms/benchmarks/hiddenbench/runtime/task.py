"""
Task definitions for HiddenBench.

A task represents a Hidden Profile scenario where information is distributed
asymmetrically across agents, and collective reasoning is required to
reach the correct answer.

Supports both:
- Official HiddenBench format (from HuggingFace: YuxuanLi1225/HiddenBench)
- Custom format with pre-divided agent information
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TaskInfo:
    """Information piece in a task."""
    content: str
    is_shared: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"content": self.content, "is_shared": self.is_shared}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskInfo":
        return cls(content=data["content"], is_shared=data.get("is_shared", False))


@dataclass
class Task:
    """
    A Hidden Profile task for the benchmark.

    Attributes:
        id: Unique identifier for the task.
        name: Human-readable name.
        description: Task description/scenario presented to agents.
        options: List of answer options to choose from.
        correct_answer: The correct option.
        shared_info: Information available to all agents.
        unshared_info: List of information sets, one per agent.
        hidden_info_pool: Optional flat list of hidden info (official format).
    """
    id: str
    name: str
    description: str
    options: list[str]
    correct_answer: str
    shared_info: list[TaskInfo]
    unshared_info: list[list[TaskInfo]]
    metadata: dict[str, Any] = field(default_factory=dict)
    # For official HiddenBench format - flat pool of hidden info to distribute
    hidden_info_pool: list[TaskInfo] | None = None

    def distribute_hidden_info(self, num_agents: int, seed: int | None = None) -> None:
        """
        Distribute hidden_info_pool across agents.

        This is used when loading official HiddenBench format where hidden
        information is provided as a flat list rather than pre-divided.

        Args:
            num_agents: Number of agents to distribute information to.
            seed: Optional random seed for reproducibility.
        """
        if self.hidden_info_pool is None:
            return

        if seed is not None:
            random.seed(seed)

        # Shuffle the pool
        pool = list(self.hidden_info_pool)
        random.shuffle(pool)

        # Distribute round-robin style
        self.unshared_info = [[] for _ in range(num_agents)]
        for i, info in enumerate(pool):
            agent_idx = i % num_agents
            self.unshared_info[agent_idx].append(info)

    def get_agent_info(self, agent_index: int, shuffle: bool = True) -> list[TaskInfo]:
        """
        Get the information set for a specific agent.

        Args:
            agent_index: Index of the agent (0-based).
            shuffle: Whether to shuffle the information order.

        Returns:
            Combined list of shared and agent-specific unshared information.
        """
        if agent_index >= len(self.unshared_info):
            raise ValueError(
                f"Agent index {agent_index} out of range. "
                f"Task has {len(self.unshared_info)} agent info sets."
            )

        # Combine shared and unshared info
        info = list(self.shared_info) + list(self.unshared_info[agent_index])

        if shuffle:
            random.shuffle(info)

        return info

    def get_full_info(self, shuffle: bool = True) -> list[TaskInfo]:
        """
        Get all information (Full Profile condition).

        Args:
            shuffle: Whether to shuffle the information order.

        Returns:
            Complete list of all information.
        """
        info = list(self.shared_info)
        for agent_info in self.unshared_info:
            info.extend(agent_info)

        if shuffle:
            random.shuffle(info)

        return info

    def format_info_for_prompt(self, info: list[TaskInfo]) -> str:
        """Format information list for inclusion in a prompt."""
        lines = []
        for i, item in enumerate(info, 1):
            lines.append(f"{i}. {item.content}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert task to dictionary."""
        result = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "options": self.options,
            "correct_answer": self.correct_answer,
            "shared_info": [i.to_dict() for i in self.shared_info],
            "unshared_info": [
                [i.to_dict() for i in agent_info]
                for agent_info in self.unshared_info
            ],
            "metadata": self.metadata,
        }
        if self.hidden_info_pool is not None:
            result["hidden_info_pool"] = [i.to_dict() for i in self.hidden_info_pool]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        """Create task from dictionary (custom format)."""
        return cls(
            id=str(data["id"]),
            name=data["name"],
            description=data["description"],
            options=data["options"],
            correct_answer=data["correct_answer"],
            shared_info=[TaskInfo.from_dict(i) for i in data["shared_info"]],
            unshared_info=[
                [TaskInfo.from_dict(i) for i in agent_info]
                for agent_info in data["unshared_info"]
            ],
            metadata=data.get("metadata", {}),
            hidden_info_pool=(
                [TaskInfo.from_dict(i) for i in data["hidden_info_pool"]]
                if "hidden_info_pool" in data else None
            ),
        )

    @classmethod
    def from_official_format(cls, data: dict[str, Any], num_agents: int = 3) -> "Task":
        """
        Create task from official HiddenBench format.

        Official format has:
        - id: int
        - name: str
        - description: str
        - shared_information: list[str]
        - hidden_information: list[str] (flat list, not divided per agent)
        - possible_answers: list[str]
        - correct_answer: str

        Args:
            data: Dictionary in official format.
            num_agents: Number of agents to distribute hidden info to.

        Returns:
            Task instance with hidden info distributed across agents.
        """
        # Create TaskInfo objects for shared info
        shared_info = [
            TaskInfo(content=info, is_shared=True)
            for info in data.get("shared_information", [])
        ]

        # Create TaskInfo objects for hidden info (pool)
        hidden_pool = [
            TaskInfo(content=info, is_shared=False)
            for info in data.get("hidden_information", [])
        ]

        task = cls(
            id=str(data["id"]),
            name=data["name"],
            description=data["description"],
            options=data.get("possible_answers", []),
            correct_answer=data["correct_answer"],
            shared_info=shared_info,
            unshared_info=[[] for _ in range(num_agents)],  # Will be populated
            hidden_info_pool=hidden_pool,
            metadata={
                "source": "official_hiddenbench",
                "original_id": data["id"],
            },
        )

        # Distribute hidden info across agents
        task.distribute_hidden_info(num_agents)

        return task

    @classmethod
    def from_file(cls, path: str | Path, num_agents: int = 3) -> "Task":
        """
        Load task from JSON file.

        Automatically detects format (official vs custom).

        Args:
            path: Path to JSON file.
            num_agents: Number of agents (used for official format).

        Returns:
            Task instance.
        """
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)

        # Detect format based on keys
        if "shared_information" in data or "hidden_information" in data:
            # Official HiddenBench format
            return cls.from_official_format(data, num_agents)
        else:
            # Custom format
            return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        """Save task to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class AgentDecision:
    """A decision made by an agent."""
    agent_id: int
    vote: str
    rationale: str
    is_correct: bool
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "vote": self.vote,
            "rationale": self.rationale,
            "is_correct": self.is_correct,
            "raw_response": self.raw_response,
        }


@dataclass
class DiscussionMessage:
    """A message in the discussion phase."""
    round_num: int
    agent_id: int
    content: str
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_num,
            "agent_id": self.agent_id,
            "content": self.content,
            "timestamp": self.timestamp,
        }


@dataclass
class TaskResult:
    """
    Complete result for a single task evaluation.

    Attributes:
        task: The task that was evaluated.
        pre_discussion_decisions: Decisions before discussion.
        post_discussion_decisions: Decisions after discussion.
        discussion_history: All messages from the discussion.
        full_profile_decisions: Decisions with full information (baseline).
        consensus_round: Round when unanimous consensus was first reached (None if never).
        stopped_early: Whether the discussion stopped early due to consensus.
        total_rounds: Total rounds that were actually run.
        agent_initial_prompts: Initial scenario prompts shown to each agent.
        pre_accuracy: Pre-discussion accuracy (average rule).
        post_accuracy: Post-discussion accuracy (average rule).
        full_profile_accuracy: Full Profile accuracy (baseline).
        pre_majority_correct: Whether majority was correct pre-discussion.
        post_majority_correct: Whether majority was correct post-discussion.
    """
    task: Task
    pre_discussion_decisions: list[AgentDecision]
    post_discussion_decisions: list[AgentDecision]
    discussion_history: list[DiscussionMessage]
    full_profile_decisions: list[AgentDecision] | None = None
    consensus_round: int | None = None  # Round when consensus was first reached (0-indexed)
    stopped_early: bool = False  # Whether discussion stopped early due to consensus
    total_rounds: int = 0  # Total rounds actually run
    agent_initial_prompts: list[str] | None = None  # Initial prompts per agent
    # Per-task resource accounting (for compression experiments)
    input_tokens: int = 0
    output_tokens: int = 0
    wall_time_seconds: float = 0.0
    channel_stats: dict[str, int] | None = None

    @property
    def pre_accuracy(self) -> float:
        """Calculate pre-discussion accuracy (average rule)."""
        if not self.pre_discussion_decisions:
            return 0.0
        correct = sum(1 for d in self.pre_discussion_decisions if d.is_correct)
        return correct / len(self.pre_discussion_decisions)

    @property
    def post_accuracy(self) -> float:
        """Calculate post-discussion accuracy (average rule)."""
        if not self.post_discussion_decisions:
            return 0.0
        correct = sum(1 for d in self.post_discussion_decisions if d.is_correct)
        return correct / len(self.post_discussion_decisions)

    @property
    def full_profile_accuracy(self) -> float | None:
        """Calculate Full Profile accuracy (baseline)."""
        if not self.full_profile_decisions:
            return None
        correct = sum(1 for d in self.full_profile_decisions if d.is_correct)
        return correct / len(self.full_profile_decisions)

    @property
    def pre_majority_correct(self) -> bool:
        """Check if majority was correct pre-discussion."""
        if not self.pre_discussion_decisions:
            return False
        correct = sum(1 for d in self.pre_discussion_decisions if d.is_correct)
        return correct > len(self.pre_discussion_decisions) / 2

    @property
    def post_majority_correct(self) -> bool:
        """Check if majority was correct post-discussion."""
        if not self.post_discussion_decisions:
            return False
        correct = sum(1 for d in self.post_discussion_decisions if d.is_correct)
        return correct > len(self.post_discussion_decisions) / 2

    @property
    def information_integration_gain(self) -> float:
        """Calculate the gain from discussion (post - pre accuracy)."""
        return self.post_accuracy - self.pre_accuracy

    @property
    def collective_reasoning_gap(self) -> float | None:
        """Calculate gap between full profile and post-discussion."""
        if self.full_profile_accuracy is None:
            return None
        return self.full_profile_accuracy - self.post_accuracy

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "task": self.task.to_dict(),
            "pre_discussion_decisions": [d.to_dict() for d in self.pre_discussion_decisions],
            "post_discussion_decisions": [d.to_dict() for d in self.post_discussion_decisions],
            "discussion_history": [m.to_dict() for m in self.discussion_history],
            "full_profile_decisions": (
                [d.to_dict() for d in self.full_profile_decisions]
                if self.full_profile_decisions else None
            ),
            "consensus_round": self.consensus_round,
            "stopped_early": self.stopped_early,
            "total_rounds": self.total_rounds,
            "agent_initial_prompts": self.agent_initial_prompts,
            "resources": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "wall_time_seconds": self.wall_time_seconds,
                "channel_stats": self.channel_stats,
            },
            "metrics": {
                "pre_accuracy": self.pre_accuracy,
                "post_accuracy": self.post_accuracy,
                "full_profile_accuracy": self.full_profile_accuracy,
                "pre_majority_correct": self.pre_majority_correct,
                "post_majority_correct": self.post_majority_correct,
                "information_integration_gain": self.information_integration_gain,
                "collective_reasoning_gap": self.collective_reasoning_gap,
            },
        }


def load_tasks_from_directory(
    directory: str | Path,
    num_agents: int = 3,
) -> list[Task]:
    """
    Load all tasks from a directory.

    Supports both individual task JSON files and the official HiddenBench
    benchmark.json format.

    Args:
        directory: Path to directory containing task JSON files.
        num_agents: Number of agents (for distributing hidden info).

    Returns:
        List of loaded tasks.
    """
    directory = Path(directory)
    tasks = []

    if not directory.exists():
        return tasks

    # Check for official benchmark.json file first
    benchmark_file = directory / "benchmark.json"
    if benchmark_file.exists():
        try:
            loaded = load_official_benchmark(benchmark_file, num_agents)
            tasks.extend(loaded)
        except Exception as e:
            print(f"Warning: Failed to load benchmark.json: {e}")

    # Also load individual task files
    for path in directory.glob("*.json"):
        if path.name == "benchmark.json":
            continue  # Already handled above
        try:
            task = Task.from_file(path, num_agents)
            tasks.append(task)
        except Exception as e:
            print(f"Warning: Failed to load task from {path}: {e}")

    return tasks


def load_official_benchmark(
    path: str | Path,
    num_agents: int = 3,
) -> list[Task]:
    """
    Load tasks from official HiddenBench benchmark.json file.

    The official file contains a JSON array of tasks in the format:
    {
        "id": int,
        "name": str,
        "description": str,
        "shared_information": list[str],
        "hidden_information": list[str],
        "possible_answers": list[str],
        "correct_answer": str
    }

    Args:
        path: Path to benchmark.json file.
        num_agents: Number of agents to distribute hidden info to.

    Returns:
        List of Task objects.
    """
    path = Path(path)
    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("benchmark.json must contain a JSON array of tasks")

    tasks = []
    for item in data:
        try:
            task = Task.from_official_format(item, num_agents)
            tasks.append(task)
        except Exception as e:
            print(f"Warning: Failed to load task {item.get('id', 'unknown')}: {e}")

    return tasks


def get_task_summary(tasks: list[Task]) -> dict[str, Any]:
    """
    Generate a summary of loaded tasks.

    Args:
        tasks: List of tasks to summarize.

    Returns:
        Dictionary with summary statistics.
    """
    if not tasks:
        return {
            "total_tasks": 0,
            "sources": {},
            "option_counts": {},
        }

    sources: dict[str, int] = {}
    option_counts: dict[int, int] = {}

    for task in tasks:
        # Track sources
        source = task.metadata.get("source", "custom")
        sources[source] = sources.get(source, 0) + 1

        # Track option counts
        n_options = len(task.options)
        option_counts[n_options] = option_counts.get(n_options, 0) + 1

    return {
        "total_tasks": len(tasks),
        "sources": sources,
        "option_counts": option_counts,
        "task_ids": [t.id for t in tasks],
        "task_names": [t.name for t in tasks],
    }

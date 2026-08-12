"""
Core HiddenBench evaluation logic.

Implements the Hidden Profile paradigm for evaluating collective reasoning
in multi-agent LLM systems.
"""

import json
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from .config import Config

# Progress callback type: (phase, current_step, total_steps, detail_message, tokens_used)
# tokens_used is a dict with "input", "output", "total" keys
ProgressCallback = Callable[[str, int, int, str, dict[str, int]], None]

# Status callback for rate limiting and other status messages
# (message_type, message) where message_type is "rate_limit", "retry", "info", etc.
StatusCallback = Callable[[str, str], None]


class RateLimitError(Exception):
    """Raised when an API rate limit is hit."""
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


def _is_rate_limit_error(error: Exception) -> tuple[bool, float | None]:
    """
    Check if an exception is a rate limit error from any provider.

    Returns:
        Tuple of (is_rate_limit, suggested_wait_time).
    """
    error_str = str(error).lower()

    # Check for common rate limit indicators
    rate_limit_indicators = [
        "rate_limit",
        "rate limit",
        "ratelimit",
        "too many requests",
        "429",
        "quota exceeded",
        "tokens per minute",
        "requests per minute",
    ]

    is_rate_limit = any(indicator in error_str for indicator in rate_limit_indicators)

    # Try to extract retry-after time if present
    retry_after = None
    if is_rate_limit:
        # Default wait time for rate limits (60 seconds)
        retry_after = 60.0

        # Try to find a more specific wait time in the error message
        import re
        # Look for patterns like "retry after 30 seconds" or "wait 60s"
        time_match = re.search(r'(?:retry|wait|after)\s*(?:in\s*)?(\d+)\s*(?:s|sec|second)', error_str)
        if time_match:
            retry_after = float(time_match.group(1))

    return is_rate_limit, retry_after
from .providers.base import LLMProvider, Message, Response
from .providers.factory import create_provider
from .task import (
    Task,
    TaskResult,
    AgentDecision,
    DiscussionMessage,
    load_tasks_from_directory,
)
from .channel import Channel, ChannelStats, build_channel
from .prompts import (
    SYSTEM_PROMPT,
    FIRST_SPEAKER_PROMPT,
    get_first_speaker_prompt,
    format_scenario_prompt,
    format_subsequent_speaker_prompt,
    format_final_decision_prompt,
    format_pre_discussion_prompt,
    format_full_profile_prompt,
)


@dataclass
class AgentState:
    """State for a single agent during evaluation."""
    agent_id: int
    messages: list[Message] = field(default_factory=list)
    info_text: str = ""
    # Number of discussion messages this agent has already been shown
    # (used by the "delta" channel protocol to avoid retransmission)
    seen_messages: int = 0

    def add_message(self, message: Message) -> None:
        """Add a message to the agent's conversation history."""
        self.messages.append(message)


def parse_decision(response: str, options: list[str]) -> tuple[str, str]:
    """
    Parse a decision from LLM response.

    Args:
        response: Raw LLM response.
        options: Valid options to choose from.

    Returns:
        Tuple of (vote, rationale).
    """
    # Try to parse as JSON
    try:
        # Find JSON in the response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            vote = data.get("vote", "").strip()
            rationale = data.get("rationale", "").strip()

            # Normalize the vote to match options
            vote_lower = vote.lower()
            for option in options:
                if option.lower() == vote_lower or option.lower() in vote_lower:
                    return option, rationale

            return vote, rationale
    except json.JSONDecodeError:
        pass

    # Fallback: try to find an option mentioned in the response
    response_lower = response.lower()
    for option in options:
        if option.lower() in response_lower:
            return option, response

    return "", response


class HiddenBench:
    """
    HiddenBench evaluation framework.

    Evaluates collective reasoning in multi-agent LLM systems using
    the Hidden Profile paradigm.
    """

    def __init__(
        self,
        config: Config,
        provider: LLMProvider | None = None,
        channel: Channel | None = None,
    ):
        """
        Initialize HiddenBench.

        Args:
            config: Configuration object.
            provider: Optional pre-configured LLM provider.
            channel: Optional pre-configured communication channel. Defaults
                to the channel described by config.channel (which itself
                defaults to the original full-history, uncompressed protocol).
        """
        self.config = config
        self._provider = provider
        self.channel = channel or build_channel(
            protocol=config.channel.protocol,
            compressor=config.channel.compressor,
            **config.channel.compressor_params(),
        )
        self._message_style = config.channel.message_style
        self._results: list[TaskResult] = []
        # Token usage tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        # Status callback for rate limits and other messages
        self._status_callback: StatusCallback | None = None

    @property
    def provider(self) -> LLMProvider:
        """Get or create the LLM provider."""
        if self._provider is None:
            provider_name = self.config.benchmark.provider
            provider_config = self.config.get_provider_config(provider_name)
            self._provider = create_provider(provider_name, provider_config)

            # Set model if specified
            if self.config.benchmark.model:
                self._provider.model = self.config.benchmark.model

            # Validate the provider
            self._provider.validate_config()

        return self._provider

    def set_status_callback(self, callback: StatusCallback | None) -> None:
        """Set the status callback for rate limit and other status messages."""
        self._status_callback = callback

    def _call_llm(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> Response:
        """
        Make an LLM call with retry logic and rate limit handling.

        Args:
            messages: Conversation messages.
            **kwargs: Additional parameters.

        Returns:
            Full Response object including token usage.
        """
        max_retries = self.config.advanced.max_retries
        retry_delay = self.config.advanced.retry_delay
        max_rate_limit_retries = 10  # More retries for rate limits
        rate_limit_retry_count = 0

        attempt = 0
        while True:
            try:
                response = self.provider.complete(
                    messages,
                    temperature=self.config.benchmark.temperature,
                    max_tokens=self.config.benchmark.max_tokens,
                    **kwargs,
                )
                # Track token usage
                self._total_input_tokens += response.input_tokens
                self._total_output_tokens += response.output_tokens
                return response
            except Exception as e:
                # Check if this is a rate limit error
                is_rate_limit, suggested_wait = _is_rate_limit_error(e)

                if is_rate_limit and rate_limit_retry_count < max_rate_limit_retries:
                    rate_limit_retry_count += 1
                    wait_time = suggested_wait or 60.0

                    # Notify via callback
                    if self._status_callback:
                        self._status_callback(
                            "rate_limit",
                            f"Rate limit hit. Waiting {wait_time:.0f}s before retry {rate_limit_retry_count}/{max_rate_limit_retries}..."
                        )

                    time.sleep(wait_time)
                    continue  # Retry without incrementing normal attempt counter

                # Not a rate limit error, or we've exhausted rate limit retries
                attempt += 1
                if attempt < max_retries:
                    wait_time = retry_delay * attempt
                    if self._status_callback:
                        self._status_callback(
                            "retry",
                            f"LLM call failed, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})..."
                        )
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {e}")

        return Response(content="", model="")

    def _initialize_agents(
        self,
        task: Task,
        num_agents: int,
    ) -> tuple[list[AgentState], list[str]]:
        """
        Initialize agent states for a task.

        Args:
            task: The task to evaluate.
            num_agents: Number of agents.

        Returns:
            Tuple of (list of agent states, list of initial prompts per agent).
        """
        agents = []
        initial_prompts = []

        for i in range(num_agents):
            # Get agent's information set
            agent_info = task.get_agent_info(i, shuffle=True)
            info_text = task.format_info_for_prompt(agent_info)

            # Build the scenario prompt
            scenario_prompt = format_scenario_prompt(
                description=task.description,
                information=info_text,
                options=task.options,
            )

            # Create initial messages
            system_text = SYSTEM_PROMPT + self.channel.get_system_prompt_suffix()
            messages = [
                Message.system(system_text),
                Message.user(scenario_prompt),
            ]

            agents.append(AgentState(
                agent_id=i,
                messages=messages,
                info_text=info_text,
            ))

            # Capture the full initial prompt (system + scenario)
            initial_prompts.append(f"[System Prompt]\n{SYSTEM_PROMPT}\n\n[Scenario]\n{scenario_prompt}")

        return agents, initial_prompts

    def _get_pre_discussion_decisions(
        self,
        agents: list[AgentState],
        task: Task,
        progress_callback: ProgressCallback | None = None,
    ) -> list[AgentDecision]:
        """
        Get pre-discussion decisions from all agents.

        Args:
            agents: List of agent states.
            task: The task being evaluated.

        Returns:
            List of pre-discussion decisions.
        """
        decisions = []
        total_agents = len(agents)

        for i, agent in enumerate(agents):
            # Create temporary messages for pre-discussion decision
            messages = agent.messages.copy()
            messages.append(Message.user(format_pre_discussion_prompt(task.options)))

            # Get decision
            response = self._call_llm(messages)
            vote, rationale = parse_decision(response.content, task.options)

            if progress_callback:
                progress_callback(
                    "pre_discussion", i + 1, total_agents, f"Agent {agent.agent_id + 1}",
                    {"input": self._total_input_tokens, "output": self._total_output_tokens, "total": self.total_tokens}
                )

            decisions.append(AgentDecision(
                agent_id=agent.agent_id,
                vote=vote,
                rationale=rationale,
                is_correct=(vote == task.correct_answer),
                raw_response=response.content,
            ))

        return decisions

    def _detect_consensus_from_messages(
        self,
        discussion_history: list[DiscussionMessage],
        task: Task,
        num_agents: int,
        round_num: int,
    ) -> str | None:
        """
        Detect if all agents have reached consensus on an answer in a given round.

        Parses the most recent message from each agent and checks if they all
        mention the same option as their preferred choice.

        Args:
            discussion_history: All discussion messages so far.
            task: The task being evaluated (for options list).
            num_agents: Number of agents.
            round_num: The round to check for consensus.

        Returns:
            The consensus vote if all agents agree, None otherwise.
        """
        # Get the last message from each agent in this round
        round_messages = [m for m in discussion_history if m.round_num == round_num]

        if len(round_messages) < num_agents:
            return None  # Not all agents have spoken yet

        # Try to parse each agent's preference from their message
        agent_preferences: dict[int, str | None] = {}

        for msg in round_messages:
            content_lower = msg.content.lower()
            detected_option = None

            # If the message is structured (schema message style), read the
            # agent's stated preference directly
            json_match = re.search(r'\{.*\}', msg.content, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    stated = str(data.get("lean") or data.get("vote") or "").strip().lower()
                    if stated:
                        for option in task.options:
                            if option.lower() == stated or option.lower() in stated:
                                detected_option = option
                                break
                except (json.JSONDecodeError, AttributeError):
                    pass

            # Look for explicit mentions of options
            for option in task.options:
                if detected_option:
                    break
                option_lower = option.lower()
                # Check for various patterns indicating preference
                if any(pattern in content_lower for pattern in [
                    f"vote for {option_lower}",
                    f"choose {option_lower}",
                    f"support {option_lower}",
                    f"agree on {option_lower}",
                    f"go with {option_lower}",
                    f"prefer {option_lower}",
                    f"decision is {option_lower}",
                    f"answer is {option_lower}",
                    f"should be {option_lower}",
                    f"it's {option_lower}",
                    f"it is {option_lower}",
                ]):
                    detected_option = option
                    break

            # Fallback: just check if only one option is mentioned
            if detected_option is None:
                mentioned_options = [opt for opt in task.options if opt.lower() in content_lower]
                if len(mentioned_options) == 1:
                    detected_option = mentioned_options[0]

            agent_preferences[msg.agent_id] = detected_option

        # Check if all agents have the same non-None preference
        preferences = list(agent_preferences.values())
        if None in preferences:
            return None

        if len(set(preferences)) == 1:
            return preferences[0]

        return None

    def _run_discussion(
        self,
        agents: list[AgentState],
        task: Task,
        num_rounds: int,
        progress_callback: ProgressCallback | None = None,
        early_stop_on_consensus: bool = True,
    ) -> tuple[list[DiscussionMessage], int | None, bool, int]:
        """
        Run the discussion phase.

        Args:
            agents: List of agent states.
            task: The task being evaluated.
            num_rounds: Number of discussion rounds.
            progress_callback: Optional callback for progress updates.
            early_stop_on_consensus: Whether to stop early if consensus is reached.

        Returns:
            Tuple of (discussion_history, consensus_round, stopped_early, total_rounds_run).
        """
        discussion_history: list[DiscussionMessage] = []
        num_agents = len(agents)
        total_messages = num_rounds * num_agents
        current_message = 0
        consensus_round: int | None = None
        stopped_early = False
        actual_rounds = 0

        for round_num in range(num_rounds):
            actual_rounds = round_num + 1

            for agent in agents:
                current_message += 1
                timestamp = datetime.now().isoformat()

                # Determine the prompt based on position in discussion
                if round_num == 0 and agent.agent_id == 0:
                    # First speaker
                    agent.add_message(Message.user(
                        get_first_speaker_prompt(self._message_style)
                    ))
                else:
                    # Subsequent speaker - show previous messages, as selected
                    # and compressed by the communication channel
                    view = self.channel.discussion_view(
                        discussion_history, agent.agent_id, agent.seen_messages
                    )
                    agent.add_message(Message.user(
                        format_subsequent_speaker_prompt(view, self._message_style)
                    ))

                # Get agent's contribution
                response = self._call_llm(agent.messages)
                self.channel.record_sent(response.content)

                if progress_callback:
                    progress_callback(
                        "discussion",
                        current_message,
                        total_messages,
                        f"Round {round_num + 1}/{num_rounds}, Agent {agent.agent_id + 1}",
                        {"input": self._total_input_tokens, "output": self._total_output_tokens, "total": self.total_tokens}
                    )

                # Record the response
                agent.add_message(Message.assistant(response.content))

                discussion_history.append(DiscussionMessage(
                    round_num=round_num,
                    agent_id=agent.agent_id,
                    content=response.content,
                    timestamp=timestamp,
                ))
                # The agent has now seen everything up to and including its
                # own message
                agent.seen_messages = len(discussion_history)

            # After each round, check for consensus
            consensus_vote = self._detect_consensus_from_messages(
                discussion_history, task, num_agents, round_num
            )

            if consensus_vote is not None and consensus_round is None:
                consensus_round = round_num  # 0-indexed

                if early_stop_on_consensus:
                    stopped_early = True
                    if self._status_callback:
                        self._status_callback(
                            "info",
                            f"Consensus reached in round {round_num + 1} on '{consensus_vote}'"
                        )
                    break

        return discussion_history, consensus_round, stopped_early, actual_rounds

    def _get_post_discussion_decisions(
        self,
        agents: list[AgentState],
        task: Task,
        discussion_history: list[DiscussionMessage],
        progress_callback: ProgressCallback | None = None,
    ) -> list[AgentDecision]:
        """
        Get post-discussion decisions from all agents.

        Args:
            agents: List of agent states.
            task: The task being evaluated.
            discussion_history: The discussion that occurred.

        Returns:
            List of post-discussion decisions.
        """
        decisions = []
        total_agents = len(agents)

        for i, agent in enumerate(agents):
            # Discussion recap for the final decision prompt, as selected and
            # compressed by the communication channel
            view = self.channel.final_view(
                discussion_history, agent.agent_id, agent.seen_messages
            )

            # Add final decision prompt
            agent.add_message(Message.user(
                format_final_decision_prompt(view, task.options)
            ))

            # Get decision
            response = self._call_llm(agent.messages)
            vote, rationale = parse_decision(response.content, task.options)

            if progress_callback:
                progress_callback(
                    "post_discussion", i + 1, total_agents, f"Agent {agent.agent_id + 1}",
                    {"input": self._total_input_tokens, "output": self._total_output_tokens, "total": self.total_tokens}
                )

            agent.add_message(Message.assistant(response.content))

            decisions.append(AgentDecision(
                agent_id=agent.agent_id,
                vote=vote,
                rationale=rationale,
                is_correct=(vote == task.correct_answer),
                raw_response=response.content,
            ))

        return decisions

    def _run_full_profile(
        self,
        task: Task,
        num_agents: int,
        progress_callback: ProgressCallback | None = None,
    ) -> list[AgentDecision]:
        """
        Run Full Profile baseline evaluation.

        Each agent receives ALL information and makes an independent decision.

        Args:
            task: The task to evaluate.
            num_agents: Number of agents.

        Returns:
            List of Full Profile decisions.
        """
        decisions = []

        # Get complete information
        full_info = task.get_full_info(shuffle=True)
        info_text = task.format_info_for_prompt(full_info)

        for agent_id in range(num_agents):
            system_text = SYSTEM_PROMPT + self.channel.get_system_prompt_suffix()
            messages = [
                Message.system(system_text),
                Message.user(format_full_profile_prompt(
                    description=task.description,
                    information=info_text,
                    options=task.options,
                )),
            ]

            response = self._call_llm(messages)
            vote, rationale = parse_decision(response.content, task.options)

            if progress_callback:
                progress_callback(
                    "full_profile", agent_id + 1, num_agents, f"Agent {agent_id + 1}",
                    {"input": self._total_input_tokens, "output": self._total_output_tokens, "total": self.total_tokens}
                )

            decisions.append(AgentDecision(
                agent_id=agent_id,
                vote=vote,
                rationale=rationale,
                is_correct=(vote == task.correct_answer),
                raw_response=response.content,
            ))

        return decisions

    def evaluate_task(
        self,
        task: Task,
        num_agents: int | None = None,
        num_rounds: int | None = None,
        run_full_profile: bool | None = None,
        early_stop_on_consensus: bool = True,
        progress_callback: ProgressCallback | None = None,
    ) -> TaskResult:
        """
        Evaluate a single task.

        Args:
            task: The task to evaluate.
            num_agents: Number of agents (default from config).
            num_rounds: Number of discussion rounds (default from config).
            run_full_profile: Whether to run Full Profile baseline.
            early_stop_on_consensus: Whether to stop discussion early if consensus reached.
            progress_callback: Optional callback for progress updates.

        Returns:
            TaskResult containing all evaluation data.
        """
        num_agents = num_agents or self.config.benchmark.num_agents
        num_rounds = num_rounds or self.config.benchmark.num_rounds
        run_full_profile = (
            run_full_profile if run_full_profile is not None
            else self.config.benchmark.run_full_profile
        )

        # Validate task has enough info sets for agents
        if len(task.unshared_info) < num_agents:
            raise ValueError(
                f"Task '{task.id}' only has {len(task.unshared_info)} "
                f"unshared info sets, but {num_agents} agents are configured."
            )

        # Set random seed if configured
        if self.config.benchmark.seed is not None:
            random.seed(self.config.benchmark.seed)

        # Per-task resource accounting
        task_start_time = time.time()
        start_input_tokens = self._total_input_tokens
        start_output_tokens = self._total_output_tokens
        channel_stats_before = self.channel.stats.snapshot()
        # Give the compressor the public answer options (protects option
        # words from deletion; options are known to every agent already)
        self.channel.set_task_context(task.options)

        # Initialize agents
        if progress_callback:
            progress_callback("init", 0, 1, "Initializing agents",
                {"input": self._total_input_tokens, "output": self._total_output_tokens, "total": self.total_tokens})
        agents, initial_prompts = self._initialize_agents(task, num_agents)

        # Phase 1: Pre-discussion decisions
        if progress_callback:
            progress_callback("pre_discussion", 0, num_agents, "Starting pre-discussion phase",
                {"input": self._total_input_tokens, "output": self._total_output_tokens, "total": self.total_tokens})
        pre_decisions = self._get_pre_discussion_decisions(agents, task, progress_callback)

        # Phase 2: Discussion
        if progress_callback:
            total_messages = num_rounds * num_agents
            progress_callback("discussion", 0, total_messages, "Starting discussion phase",
                {"input": self._total_input_tokens, "output": self._total_output_tokens, "total": self.total_tokens})
        discussion_history, consensus_round, stopped_early, total_rounds = self._run_discussion(
            agents, task, num_rounds, progress_callback, early_stop_on_consensus
        )

        # Phase 3: Post-discussion decisions
        if progress_callback:
            progress_callback("post_discussion", 0, num_agents, "Starting post-discussion phase",
                {"input": self._total_input_tokens, "output": self._total_output_tokens, "total": self.total_tokens})
        post_decisions = self._get_post_discussion_decisions(
            agents, task, discussion_history, progress_callback
        )

        # Phase 4: Full Profile baseline (optional)
        full_profile_decisions = None
        if run_full_profile:
            if progress_callback:
                progress_callback("full_profile", 0, num_agents, "Starting Full Profile baseline",
                    {"input": self._total_input_tokens, "output": self._total_output_tokens, "total": self.total_tokens})
            full_profile_decisions = self._run_full_profile(task, num_agents, progress_callback)

        result = TaskResult(
            task=task,
            pre_discussion_decisions=pre_decisions,
            post_discussion_decisions=post_decisions,
            discussion_history=discussion_history,
            full_profile_decisions=full_profile_decisions,
            consensus_round=consensus_round,
            stopped_early=stopped_early,
            total_rounds=total_rounds,
            agent_initial_prompts=initial_prompts,
            input_tokens=self._total_input_tokens - start_input_tokens,
            output_tokens=self._total_output_tokens - start_output_tokens,
            wall_time_seconds=time.time() - task_start_time,
            channel_stats=ChannelStats.delta(
                channel_stats_before, self.channel.stats.snapshot()
            ),
        )

        self._results.append(result)
        return result

    def run(
        self,
        tasks: list[Task] | None = None,
        tasks_dir: str | None = None,
    ) -> list[TaskResult]:
        """
        Run the benchmark on multiple tasks.

        Args:
            tasks: List of tasks to evaluate.
            tasks_dir: Directory to load tasks from.

        Returns:
            List of task results.
        """
        if tasks is None:
            tasks_dir = tasks_dir or self.config.benchmark.tasks_dir
            tasks = load_tasks_from_directory(tasks_dir)

        if not tasks:
            raise ValueError(
                f"No tasks found. Please add task files to {tasks_dir} "
                "or provide tasks directly."
            )

        results = []
        for task in tasks:
            result = self.evaluate_task(task)
            results.append(result)

        return results

    @property
    def results(self) -> list[TaskResult]:
        """Get all collected results."""
        return self._results

    @property
    def total_tokens(self) -> int:
        """Get total tokens used (input + output)."""
        return self._total_input_tokens + self._total_output_tokens

    @property
    def input_tokens(self) -> int:
        """Get total input tokens used."""
        return self._total_input_tokens

    @property
    def output_tokens(self) -> int:
        """Get total output tokens used."""
        return self._total_output_tokens

    def get_token_usage(self) -> dict[str, int]:
        """Get token usage as a dictionary."""
        return {
            "input": self._total_input_tokens,
            "output": self._total_output_tokens,
            "total": self.total_tokens,
        }

    def clear_results(self) -> None:
        """Clear collected results and reset token counts."""
        self._results = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0

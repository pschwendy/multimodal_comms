"""
Prompt templates for HiddenBench.

Based on the prompts described in the HiddenBench paper.
"""

from typing import Any

from jinja2 import Template


# System prompt for agents
SYSTEM_PROMPT = """You are participating in a group decision-making task. You have received some information about a scenario and need to make a decision.

IMPORTANT: The order of the facts you receive is randomly shuffled. The order does not indicate importance or relationship between facts. Please reason carefully about all the information.

You will need to:
1. Carefully consider all the information provided to you
2. Share your thoughts concisely with the group (one or two sentences)
3. Listen to what others share
4. Make a final decision

When making your decision, provide your response in JSON format:
{
  "vote": "<your chosen option>",
  "rationale": "<brief explanation of your reasoning>"
}"""


# Prompt for presenting the scenario and information
SCENARIO_PROMPT = Template("""## Scenario
{{ description }}

## Your Information
{{ information }}

## Available Options
{% for option in options %}
- {{ option }}
{% endfor %}

Please consider this information carefully before the group discussion begins.""")


# Prompt for first speaker in discussion
FIRST_SPEAKER_PROMPT = """You are the first to speak.

Based on the information you have, share your initial thoughts with the group in one or two sentences. Focus on the key facts that seem most relevant to making a decision.

Respond with just your discussion message (not in JSON format yet - the final vote will come after the discussion)."""


# Sender-side message style constraints. These change only the instructions
# given to agents about the FORM of discussion messages, never the task
# content. "default" reproduces the original benchmark prompts exactly.
MESSAGE_STYLE_INSTRUCTIONS = {
    "default": "",
    "concise": (
        "\n\nIMPORTANT: Keep your message under 20 words. Telegraphic style is "
        "fine. State only facts or your current preference; no pleasantries."
    ),
    "schema": (
        "\n\nIMPORTANT: Respond ONLY with minified JSON, no other text: "
        '{"lean": "<option name, or undecided>", '
        '"new_facts": ["<only facts from your information that you have not already shared>"]}'
    ),
}


# Prompt for subsequent speakers in discussion
SUBSEQUENT_SPEAKER_PROMPT = Template("""Previous messages from the group discussion:

{% for message in messages %}
**Agent {{ message.agent_id + 1 }}**: {{ message.content }}
{% endfor %}

It's your turn to speak.

Based on your information and what others have shared, contribute to the discussion in one or two sentences. You may share new information, respond to others' points, or express agreement/disagreement.

Respond with just your discussion message (not in JSON format yet).""")


# Prompt for final decision after discussion
FINAL_DECISION_PROMPT = Template("""The group discussion has concluded.

## Discussion Summary
{% for message in messages %}
**Round {{ message.round_num }}, Agent {{ message.agent_id + 1 }}**: {{ message.content }}
{% endfor %}

Based on your original information and the group discussion, please make your final decision.

## Available Options
{% for option in options %}
- {{ option }}
{% endfor %}

Respond with your decision in JSON format:
{
  "vote": "<your chosen option - must be exactly one of the options above>",
  "rationale": "<brief explanation of your reasoning>"
}""")


# Prompt for pre-discussion decision
PRE_DISCUSSION_DECISION_PROMPT = Template("""Based solely on the information you have (before any group discussion), make an initial decision.

## Available Options
{% for option in options %}
- {{ option }}
{% endfor %}

Respond with your decision in JSON format:
{
  "vote": "<your chosen option - must be exactly one of the options above>",
  "rationale": "<brief explanation of your reasoning>"
}""")


# Prompt for Full Profile evaluation (agent has all information)
FULL_PROFILE_PROMPT = Template("""## Scenario
{{ description }}

## Complete Information
{{ information }}

## Available Options
{% for option in options %}
- {{ option }}
{% endfor %}

Based on ALL the information provided above, make your decision.

Respond with your decision in JSON format:
{
  "vote": "<your chosen option - must be exactly one of the options above>",
  "rationale": "<brief explanation of your reasoning>"
}""")


def format_scenario_prompt(
    description: str,
    information: str,
    options: list[str],
) -> str:
    """Format the scenario prompt for an agent."""
    return SCENARIO_PROMPT.render(
        description=description,
        information=information,
        options=options,
    )


def get_first_speaker_prompt(style: str = "default") -> str:
    """Get the first-speaker prompt, optionally with a message-style constraint."""
    return FIRST_SPEAKER_PROMPT + MESSAGE_STYLE_INSTRUCTIONS.get(style, "")


def format_subsequent_speaker_prompt(
    messages: list[dict[str, Any]],
    style: str = "default",
) -> str:
    """Format the prompt for a speaker after the first."""
    prompt = SUBSEQUENT_SPEAKER_PROMPT.render(messages=messages)
    return prompt + MESSAGE_STYLE_INSTRUCTIONS.get(style, "")


def format_final_decision_prompt(
    messages: list[dict[str, Any]],
    options: list[str],
) -> str:
    """Format the final decision prompt."""
    return FINAL_DECISION_PROMPT.render(
        messages=messages,
        options=options,
    )


def format_pre_discussion_prompt(options: list[str]) -> str:
    """Format the pre-discussion decision prompt."""
    return PRE_DISCUSSION_DECISION_PROMPT.render(options=options)


def format_full_profile_prompt(
    description: str,
    information: str,
    options: list[str],
) -> str:
    """Format the Full Profile evaluation prompt."""
    return FULL_PROFILE_PROMPT.render(
        description=description,
        information=information,
        options=options,
    )

"""
Task 3 - Multi-Tool Autonomous AI Agent (LangChain ReAct).

The agent decides on its own which tool to call (web search, calculator, file reader),
runs an executor loop and prints the full reasoning chain:

    Thought -> Action -> Observation -> ... -> Final Answer

Run from the repository root:
    python task3_multi_tool_agent/agent.py
    python task3_multi_tool_agent/agent.py --query "Read notes.txt and calculate budget per team member"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain.agents import AgentExecutor, create_react_agent  # noqa: E402
from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402
from langchain_core.prompts import PromptTemplate  # noqa: E402

from common.llm_factory import PROVIDERS, ConfigError, default_provider, get_chat_model  # noqa: E402
from tools import ALL_TOOLS  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):  # emojis on Windows consoles
    sys.stdout.reconfigure(encoding="utf-8")

REACT_TEMPLATE = """You are a helpful, accurate autonomous AI assistant.
Solve the user's question step by step. Choose tools ONLY when needed:
- Use web_search for current events or facts you are not sure about.
- Use calculator for ANY arithmetic - never calculate in your head.
- Use file_reader to read local files (use input 'list' to see the available files).
You may use several tools in sequence and combine their results.

You have access to the following tools:

{tools}

Use EXACTLY this format:

Question: the input question you must answer
Thought: reason about what to do next
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


class ConsoleTrace(BaseCallbackHandler):
    """Pretty-prints the agent's reasoning chain."""

    LINE = "-" * 70

    def __init__(self) -> None:
        self.step = 0

    def on_agent_action(self, action, **kwargs):
        self.step += 1
        thought = action.log.split("Action:")[0].replace("Thought:", "").strip()
        print(f"\n{self.LINE}\nSTEP {self.step}")
        print(f"💭 Thought       : {thought}")
        print(f"🛠  Action        : {action.tool}")
        print(f"⌨️  Action Input  : {action.tool_input}")

    def on_tool_end(self, output, **kwargs):
        text = str(output).strip()
        if len(text) > 700:
            text = text[:700] + " ...[truncated]"
        print(f"👁  Observation   : {text}")

    def on_agent_finish(self, finish, **kwargs):
        print(f"\n{self.LINE}\n✅ FINAL ANSWER :\n{finish.return_values.get('output', '')}\n{self.LINE}")


def build_executor(provider: str, model: str | None) -> AgentExecutor:
    llm = get_chat_model(provider, model, temperature=0)
    prompt = PromptTemplate.from_template(REACT_TEMPLATE)
    agent = create_react_agent(llm=llm, tools=ALL_TOOLS, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=False,               # we print our own clean trace
        max_iterations=8,            # guardrail against infinite loops
        handle_parsing_errors=True,  # recover if the LLM breaks the format
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-Tool Autonomous AI Agent")
    parser.add_argument("--provider", choices=PROVIDERS, default=default_provider())
    parser.add_argument("--model")
    parser.add_argument("--query", help="run a single question and exit")
    args = parser.parse_args()

    try:
        executor = build_executor(args.provider, args.model)
    except ConfigError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    def ask(question: str) -> None:
        try:
            executor.invoke({"input": question}, config={"callbacks": [ConsoleTrace()]})
        except Exception as exc:  # keep the REPL alive on API / network errors
            print(f"\n❌ Error: {exc}")

    if args.query:
        ask(args.query)
        return 0

    print("=" * 70)
    print(f"  MULTI-TOOL AUTONOMOUS AI AGENT   (provider: {args.provider})")
    print("  Tools: web_search | calculator | file_reader      Type 'exit' to quit")
    print("=" * 70)
    while True:
        try:
            question = input("\n🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit", "q"}:
            break
        if question:
            ask(question)
    print("Goodbye!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

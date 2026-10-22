"""
Task 3 - Multi-Tool Autonomous AI Agent (LangChain function-calling agent).
 
The model decides on its own which tool to call (web search, calculator, file reader).
Every tool call must carry a short `thought`, so the console shows the full reasoning chain:
 
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
 
from langchain.agents import AgentExecutor, create_tool_calling_agent  # noqa: E402
from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_core.tools import StructuredTool  # noqa: E402
from pydantic import Field, create_model  # noqa: E402
 
from common.llm_factory import (  # noqa: E402
    PROVIDERS,
    ConfigError,
    default_provider,
    get_chat_model,
    message_text,
)
from tools import ALL_TOOLS  # noqa: E402
 
if hasattr(sys.stdout, "reconfigure"):  # emojis on Windows consoles
    sys.stdout.reconfigure(encoding="utf-8")
 
SYSTEM_PROMPT = """You are a helpful, accurate autonomous AI assistant that solves multi-step problems with tools.
 
Tools:
- web_search: current events and facts you are not sure about.
- calculator: ANY arithmetic - never calculate in your head.
- file_reader: read local files (input 'list' shows the available files).
 
Rules:
1. Every tool call MUST include a short 'thought' that explains why you are calling that tool.
2. When one result depends on another, call tools one step at a time; you may chain several tools.
3. Do not call tools you do not need.
4. When you have enough information, give a clear and concise final answer."""
 
PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
 
# name of the single input of each base tool (see tools.py)
TOOL_INPUT_NAMES = {"web_search": "query", "calculator": "expression", "file_reader": "file_name"}
 
 
def with_thought(base_tool) -> StructuredTool:
    """Wrap a tool so the model must also pass a `thought` (its reasoning for this step)."""
    arg = TOOL_INPUT_NAMES[base_tool.name]
    args_schema = create_model(
        f"{base_tool.name}_input",
        thought=(str, Field(description="One short sentence: WHY you are calling this tool now.")),
        **{arg: (str, Field(description=f"Input for {base_tool.name}."))},
    )
 
    def run(thought: str, **kwargs) -> str:
     return base_tool.func(**kwargs)
 
    return StructuredTool.from_function(
        func=run,
        name=base_tool.name,
        description=base_tool.description,
        args_schema=args_schema,
    )
 
 
AGENT_TOOLS = [with_thought(t) for t in ALL_TOOLS]
 
 
class ConsoleTrace(BaseCallbackHandler):
    """Pretty-prints the agent's reasoning chain."""
 
    LINE = "-" * 70
 
    def __init__(self) -> None:
        self.step = 0
 
    def on_agent_action(self, action, **kwargs):
        self.step += 1
        tool_input = dict(action.tool_input) if isinstance(action.tool_input, dict) else {"input": action.tool_input}
        thought = tool_input.pop("thought", "") or "(no explanation given)"
        value = next(iter(tool_input.values()), "")
        print(f"\n{self.LINE}\nSTEP {self.step}")
        print(f"💭 Thought       : {thought}")
        print(f"🛠  Action        : {action.tool}")
        print(f"⌨️  Action Input  : {value}")
 
    def on_tool_end(self, output, **kwargs):
        text = str(output).strip()
        if len(text) > 700:
            text = text[:700] + " ...[truncated]"
        print(f"👁  Observation   : {text}")
 
    def on_agent_finish(self, finish, **kwargs):
        answer = message_text(finish.return_values.get("output", ""))
        print(f"\n{self.LINE}\n✅ FINAL ANSWER :\n{answer}\n{self.LINE}")
 
 
def build_executor(provider: str, model: str | None) -> AgentExecutor:
    llm = get_chat_model(provider, model, temperature=0)
    agent = create_tool_calling_agent(llm, AGENT_TOOLS, PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=AGENT_TOOLS,
        verbose=False,               # we print our own clean trace
        max_iterations=8,            # guardrail against infinite loops
        handle_parsing_errors=True,
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
 

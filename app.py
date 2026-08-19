import os
import sys
import io
import traceback
import re

from typing import TypedDict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langserve import add_routes


# ============================================================
# 1. GEMINI INITIALIZATION
# ============================================================

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set."
    )


llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key,
    temperature=0
)

llm = llm_flash


# ============================================================
# 2. STATE DEFINITION
# ============================================================

class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]


# ============================================================
# 3. HELPER FUNCTION
# ============================================================

def get_text(content) -> str:
    """
    Convert Gemini/LangChain responses into normal text.
    """

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, dict):
                text = item.get("text", "")

                if text:
                    parts.append(str(text))

            else:
                parts.append(str(item))

        return "\n".join(parts)

    return str(content)


def clean_human_text(text: str) -> str:
    """
    Remove Markdown formatting and escaped characters
    so the final output looks like normal human-readable text.
    """

    if not text:
        return ""

    text = str(text)

    # Convert literal escaped newlines into real newlines
    text = text.replace("\\n", "\n")

    # Remove Markdown headings
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)

    # Remove bold / italic Markdown
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("*", "")

    # Remove inline code formatting
    text = text.replace("```python", "")
    text = text.replace("```", "")
    text = text.replace("`", "")

    # Remove unnecessary leading bullet symbols
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.MULTILINE)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove spaces at the beginning/end of lines
    text = "\n".join(
        line.strip()
        for line in text.splitlines()
    )

    return text.strip()


# ============================================================
# 4. TOOLS
# ============================================================

@tool
def run_python_code(code: str) -> str:
    """Execute Python code and return the standard output or error trace."""

    if not isinstance(code, str):
        code = str(code)

    clean_code = (
        code
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    old_stdout = sys.stdout
    new_stdout = io.StringIO()

    sys.stdout = new_stdout

    try:

        local_scope = {}

        exec(clean_code, {}, local_scope)

        result = new_stdout.getvalue()

    except Exception:

        result = (
            "Execution Error:\n"
            + traceback.format_exc()
        )

    finally:

        sys.stdout = old_stdout

    return (
        result.strip()
        if result.strip()
        else "Success. The program produced no terminal output."
    )


@tool
def generate_test_cases(task_description: str) -> str:
    """Generate clear and human-readable test scenarios."""

    prompt = f"""
You are a Senior QA Engineer.

Create 3 to 5 specific test scenarios for this coding task:

{task_description}

Your response MUST be written as simple human-readable plain text.

Use this structure:

1. Test scenario title

Scenario: Explain what should be tested.

Expected Result: Explain what should happen.

2. Test scenario title

Scenario: Explain what should be tested.

Expected Result: Explain what should happen.

Continue until you have 3 to 5 test scenarios.

IMPORTANT RULES:

Do NOT use Markdown.
Do NOT use ** symbols.
Do NOT use # headings.
Do NOT use backticks.
Do NOT use bullet symbols.
Do NOT use code blocks.
Do NOT use escaped characters such as \\n.
Use normal paragraphs and numbered sections only.
Keep the explanation simple and easy for a student to understand.
"""

    response = llm.invoke(prompt)

    return get_text(response.content)


# ============================================================
# 5. GRAPH NODES
# ============================================================

def task_input_node(state: CrewState):

    return {
        "next_step": "developer"
    }


def real_time_developer(state: CrewState):

    print("[Developer] Writing dynamic code using LLM...")

    task = get_text(
        state["messages"][-1].content
    )

    dev_prompt = f"""
Write a clean Python script to solve this coding task:

{task}

IMPORTANT:
Return ONLY the Python code.
Do not provide explanations.
Do not use Markdown.
Do not use code fences.
"""

    response = llm_flash.invoke(dev_prompt)

    code_str = get_text(response.content)

    code_str = (
        code_str
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    print("[Developer] Generated code:")
    print(code_str)

    return {
        "code": code_str
    }


def real_time_tester(state: CrewState):

    print("[Tester] Generating dynamic tests and executing code...")

    task = get_text(
        state["messages"][-1].content
    )

    # --------------------------------------------------------
    # Generate test cases
    # --------------------------------------------------------

    test_cases = generate_test_cases.invoke(task)

    cases_str = clean_human_text(test_cases)

    # --------------------------------------------------------
    # Execute generated code
    # --------------------------------------------------------

    execution_result = run_python_code.invoke(
        {
            "code": state["code"]
        }
    )

    execution_result = clean_human_text(
        execution_result
    )

    # --------------------------------------------------------
    # Create clean human-readable report
    # --------------------------------------------------------

    report = (
        f"Execution Output\n\n"
        f"{execution_result}\n\n"
        f"Test Scenarios\n\n"
        f"{cases_str}"
    )

    return {
        "report": report
    }


def manager_decision_node(state: CrewState):

    print("[Manager] Reviewing test report...")

    return {
        "next_step": "archiver"
    }


def archiver_node(state: CrewState):

    print("[Archiver] Task completed successfully.")

    return {
        "next_step": "exit"
    }


# ============================================================
# 6. LANGGRAPH CONSTRUCTION
# ============================================================

rt_workflow = StateGraph(CrewState)


# ------------------------------------------------------------
# Add nodes
# ------------------------------------------------------------

rt_workflow.add_node(
    "task_input",
    task_input_node
)

rt_workflow.add_node(
    "developer",
    real_time_developer
)

rt_workflow.add_node(
    "tester",
    real_time_tester
)

rt_workflow.add_node(
    "manager_decision",
    manager_decision_node
)

rt_workflow.add_node(
    "archiver",
    archiver_node
)


# ------------------------------------------------------------
# START → task_input
# ------------------------------------------------------------

rt_workflow.add_edge(
    START,
    "task_input"
)


# ------------------------------------------------------------
# task_input → developer
# ------------------------------------------------------------

def route_from_input(state):

    if state.get("next_step") == "exit":
        return END

    return "developer"


rt_workflow.add_conditional_edges(
    "task_input",
    route_from_input
)


# ------------------------------------------------------------
# developer → tester
# ------------------------------------------------------------

rt_workflow.add_edge(
    "developer",
    "tester"
)


# ------------------------------------------------------------
# tester → manager
# ------------------------------------------------------------

rt_workflow.add_edge(
    "tester",
    "manager_decision"
)


# ------------------------------------------------------------
# manager → archiver
# ------------------------------------------------------------

def route_from_decision(state):

    if state.get("next_step") == "archiver":
        return "archiver"

    return "task_input"


rt_workflow.add_conditional_edges(
    "manager_decision",
    route_from_decision
)


# ------------------------------------------------------------
# archiver → END
# ------------------------------------------------------------

rt_workflow.add_edge(
    "archiver",
    END
)


# ------------------------------------------------------------
# Compile graph
# ------------------------------------------------------------

rt_app = rt_workflow.compile()


# ============================================================
# 7. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Agentic AI LangGraph System",
    description=(
        "Developer → Tester → Manager → Archiver "
        "Agentic AI pipeline"
    ),
    version="1.0.0"
)


# ============================================================
# 8. LANGSERVE INPUT MODEL
# ============================================================

class AgentInput(BaseModel):
    input: str


# ============================================================
# 9. LANGSERVE WRAPPER
# ============================================================

def run_langgraph(data):
    """
    Receives the Playground input,
    runs the LangGraph,
    and returns a clean human-readable response.
    """

    # --------------------------------------------------------
    # Get user's task
    # --------------------------------------------------------

    if isinstance(data, dict):
        task = data.get("input", "")
    else:
        task = str(data)

    task = str(task).strip()

    # --------------------------------------------------------
    # Create initial LangGraph state
    # --------------------------------------------------------

    initial_state: CrewState = {
        "messages": [
            HumanMessage(
                content=task
            )
        ],
        "next_step": "developer",
        "code": None,
        "report": None
    }

    # --------------------------------------------------------
    # Run LangGraph
    # --------------------------------------------------------

    final_state = rt_app.invoke(
        initial_state,
        config={
            "recursion_limit": 50
        }
    )

    # --------------------------------------------------------
    # Get final report
    # --------------------------------------------------------

    report = final_state.get(
        "report",
        "No report was generated."
    )

    report = clean_human_text(report)

    # --------------------------------------------------------
    # Return ONLY clean text
    # --------------------------------------------------------

    final_output = (
        f"Task\n\n"
        f"{task}\n\n"
        f"{report}"
    )

    return final_output


# ============================================================
# 10. CREATE LANGSERVE CHAIN
# ============================================================

agent_chain = (
    RunnableLambda(run_langgraph)
).with_types(
    input_type=AgentInput,
    output_type=str
)


# ============================================================
# 11. ROOT / HEALTH ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Agentic AI LangGraph API is running!",
        "status": "healthy",
        "playground": "/agent/playground/"
    }


# ============================================================
# 12. LANGSERVE PLAYGROUND
# ============================================================

add_routes(
    app,
    agent_chain,
    path="/agent",
    playground_type="default"
)


# ============================================================
# 13. RUN LOCALLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )

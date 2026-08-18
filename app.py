import os
import sys
import io
import traceback

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
# 3. TOOLS
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
        result = f"Execution Error:\n{traceback.format_exc()}"

    finally:
        sys.stdout = old_stdout

    return (
        result.strip()
        if result.strip()
        else "Success (no terminal output)"
    )


@tool
def generate_test_cases(task_description: str) -> str:
    """Generate specific test scenarios for a given coding task."""

    prompt = (
        f"You are a Senior QA Engineer. Generate 3 to 5 highly specific "
        f"test scenarios for the following coding task: "
        f"'{task_description}'.\n"
        f"Include standard cases and edge cases. "
        f"Return them as a numbered list."
    )

    response = llm.invoke(prompt)

    return (
        response.content
        if hasattr(response, "content")
        else str(response)
    )


# ============================================================
# 4. GRAPH NODES
# ============================================================

def task_input_node(state: CrewState):

    # In the web version, the task is already supplied
    # through the LangServe Playground.

    return {
        "next_step": "developer"
    }


def real_time_developer(state: CrewState):

    print("[Developer] Writing dynamic code using LLM...")

    task = state["messages"][-1].content

    dev_prompt = (
        f"Write a clean Python script to solve this: {task}. "
        f"Only return the code, no explanation or markdown formatting."
    )

    response = llm_flash.invoke(dev_prompt)

    content = response.content

    if isinstance(content, list):

        if content:
            first_item = content[0]

            if isinstance(first_item, dict):
                code_str = first_item.get("text", "")
            else:
                code_str = str(first_item)

        else:
            code_str = ""

    else:
        code_str = str(content)

    print("[Developer] Generated code:")
    print(code_str)

    return {
        "code": code_str
    }


def real_time_tester(state: CrewState):

    print("[Tester] Generating dynamic tests and executing code...")

    task = state["messages"][-1].content

    # --------------------------------------------------------
    # Generate test cases
    # --------------------------------------------------------

    test_cases = generate_test_cases.invoke(task)

    content = test_cases

    if isinstance(content, list):

        if content:
            first_item = content[0]

            if isinstance(first_item, dict):
                cases_str = first_item.get("text", "")
            else:
                cases_str = str(first_item)

        else:
            cases_str = ""

    else:
        cases_str = str(content)

    # --------------------------------------------------------
    # Execute generated code
    # --------------------------------------------------------

    execution_result = run_python_code.invoke(
        {
            "code": state["code"]
        }
    )

    # --------------------------------------------------------
    # Create report
    # --------------------------------------------------------

    report = (
        f"### EXECUTION OUTPUT:\n"
        f"{execution_result}\n\n"
        f"### TEST SCENARIOS EVALUATED:\n"
        f"{cases_str}"
    )

    return {
        "report": report
    }


def manager_decision_node(state: CrewState):

    print("[Manager] Reviewing test report...")

    # In the web version we don't use input().
    # The manager automatically completes the workflow.

    return {
        "next_step": "archiver"
    }


def archiver_node(state: CrewState):

    print("[Archiver] Task stored successfully.")

    return {
        "next_step": "exit"
    }


# ============================================================
# 5. LANGGRAPH CONSTRUCTION
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
# 6. FASTAPI APPLICATION
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
# 7. LANGSERVE INPUT MODEL
# ============================================================

class AgentInput(BaseModel):
    input: str


# ============================================================
# 8. LANGSERVE WRAPPER
# ============================================================

def run_langgraph(data):
    """
    Convert Playground input into CrewState,
    execute the LangGraph,
    and return a clean response.
    """

    # --------------------------------------------------------
    # Get user's task
    # --------------------------------------------------------

    if isinstance(data, dict):
        task = data.get("input", "")
    else:
        task = str(data)

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
    # Return final result
    # --------------------------------------------------------

    return {
        "task": task,
        "generated_code": final_state.get("code"),
        "report": final_state.get("report"),
        "status": "completed"
    }


agent_chain = (
    RunnableLambda(run_langgraph)
).with_types(
    input_type=AgentInput
)


# ============================================================
# 9. ROOT / HEALTH ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Agentic AI LangGraph API is running!",
        "status": "healthy",
        "playground": "/agent/playground/"
    }


# ============================================================
# 10. LANGSERVE PLAYGROUND
# ============================================================

add_routes(
    app,
    agent_chain,
    path="/agent",
    playground_type="default"
)


# ============================================================
# 11. RUN LOCALLY
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

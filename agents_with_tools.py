from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

# TODO: Create a simple calculator tool

def calculator(expression: str) -> str:
    """A simple calculator that can add, subtract, multiply, or divide two numbers.
    Input should be a mathematical expression like '2 + 2' or '15 / 3'."""
    try:
        # HINT: Use Python's eval() function for simple calculations
        # Your code here
        result = eval(expression)
        return str(result)
        
    except Exception as e:
        return f"Error calculating: {str(e)}"

# TODO: Create a text formatting tool
# Create a text formatting tool
def format_text(text: str) -> str:
    """Format text to uppercase, lowercase, or title case.
    Input should be in format: [format_type]: [text]
    where format_type is uppercase, lowercase, or titlecase.
    
    Examples:
    - uppercase: hello world -> HELLO WORLD
    - lowercase: HELLO WORLD -> hello world 
    - titlecase: hello world -> Hello World
    """
    try:
        # Handle the case where the entire string is passed
        if ":" in text:
            format_type, content = text.split(":", 1)
            format_type = format_type.strip().lower()
            content = content.strip()
        else:
            # If no colon, assume they want titlecase
            return f"Missing format. Example: titlecase: {text} -> {text.title()}"
            
        if format_type == "uppercase":
            return content.upper()
        elif format_type == "lowercase":
            return content.lower()
        elif format_type == "titlecase":
            return content.title()
        else:
            return f"Unknown format {format_type}. Use: uppercase, lowercase, or titlecase"
            
    except Exception as e:
        return f"Error formatting text: {str(e)}"
# TODO: Create Tool objects for our functions
# HINT: Use the Tool class to wrap the functions
tools = [
    Tool(
        name="calculator",
        func=calculator,
        description="Useful for performing simple math calculations"
    ),
    Tool(
        name="format_text",
        func=format_text,
        description="Useful for formatting text to uppercase, lowercase, or titlecase"
    )
]


# TODO: Create a simple prompt template
prompt_template = """You are a helpful assistant who can use tools to help with simple tasks.
You have access to these tools:
{tools}

The available tools are: {tool_names}
Follow this format
Question: the user's question
Thought: think about what to do
Action: the tool to use, should be one of [{tool_names}]
Action Input: the input to the tool
Observation: the result from the tool
Thought: I now know the final answer
Final Answer: your final answer to the user's question
Question: {input}
{agent_scratchpad}
"""
prompt = PromptTemplate.from_template(prompt_template)
agent = create_react_agent(
    llm=llama_llm,
    tools=tools,
    prompt=prompt
)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True
)


# Test with simple questions
test_questions = [
    "What is 25 + 63?", 
    "Can you convert 'hello world' to uppercase?",
    "Calculate 15 * 7", 
    "titlecase: langchain is awesome",
]

# TODO: Run the tests
for question in test_questions:
    print(f"\n===== Testing: {question} =====")
    result = agent_executor.invoke({"input": question})
    print(result['output'])
    

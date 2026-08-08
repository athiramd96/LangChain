from langchain.chains import LLMChain, SequentialChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Sample product reviews for testing
positive_review = """I absolutely love this coffee maker! It brews quickly and the coffee tastes amazing. 
The built-in grinder saves me so much time in the morning, and the programmable timer means 
I wake up to fresh coffee every day. Worth every penny and highly recommended to any coffee enthusiast."""

negative_review = """Disappointed with this laptop. It's constantly overheating after just 30 minutes of use, 
and the battery life is nowhere near the 8 hours advertised - I barely get 3 hours. 
The keyboard has already started sticking on several keys after just two weeks. Would not recommend to anyone."""

# Step 1: Define the prompt templates for each processing step
sentiment_template = """Analyze the sentiment of the following product review as positive, negative, or neutral.
Provide your analysis in the format: "SENTIMENT: [positive/negative/neutral]"

Review: {review}

Your analysis:
"""

summary_template = """Summarize the following product review into 3-5 key bullet points.
Each bullet point should be concise and capture an important aspect mentioned in the review.

Review: {review}
Sentiment: {sentiment}

Key points:
"""

response_template = """Write a helpful response to a customer based on their product review.
If the sentiment is positive, thank them for their feedback. If negative, express understanding 
and suggest a solution or next steps. Personalize based on the specific points they mentioned.

Review: {review}
Sentiment: {sentiment}
Key points: {summary}

Response to customer:
"""

# TODO: Create prompt templates for each step
sentiment_prompt_template = PromptTemplate(template=sentiment_template, input_variables=['review'])
summary_prompt_template = PromptTemplate(template=summary_template, input_variables=['review'])
response_prompt_template = PromptTemplate(template=response_template, input_variables=['review'])

# PART 1: Traditional Chain Approach
sentiment_chain = LLMChain(llm=llama_llm, prompt=sentiment_prompt_template, output_key='sentiment')
summary_chain = LLMChain(llm=llama_llm, prompt=summary_prompt_template, output_key='summary')
response_chain = LLMChain(llm=llama_llm, prompt=response_prompt_template, output_key='response')

# TODO: Create a SequentialChain to connect all steps
overall_chain = SequentialChain(
    # List of chains to execute in sequence
    chains=[sentiment_chain, summary_chain, response_chain],
    
    # The input variables required to start the chain sequence
    # Only 'location' is needed to begin the process
    input_variables=['review'],
    
    # The output variables to include in the final result
    # This makes the output of each chain available in the final result
    output_variables=['sentiment', 'summary', 'response'],
    
    # Whether to print detailed information about each step
    verbose=True
)

# PART 2: LCEL Approach
sentiment_chain_lcel = (
    PromptTemplate.from_template(sentiment_template)  # Format the prompt with location
    | llama_llm                                    # Send to the LLM
    | StrOutputParser()                              # Extract the string response
)

# Create the dish chain using LCEL
# This chain takes a meal name and returns a recipe
summary_chain_lcel = (
    PromptTemplate.from_template(summary_template)      # Format the prompt with meal
    | llama_llm                                    # Send to the LLM
    | StrOutputParser()                              # Extract the string response
)

# Create the time estimation chain using LCEL
# This chain takes a recipe and returns an estimated cooking time
response_chain_lcel = (
    PromptTemplate.from_template(response_template)      # Format the prompt with recipe
    | llama_llm                                    # Send to the LLM
    | StrOutputParser()                              # Extract the string response
)

overall_chain_lcel = (
    # Step 1: Generate a sentiment based on review and add it to the input dictionary
    RunnablePassthrough.assign(sentiment=lambda x: sentiment_chain_lcel.invoke({"review": x["review"]}))
    # Step 2: Generate a summary based on the sentiment and add it to the input dictionary
    | RunnablePassthrough.assign(summary=lambda x: summary_chain_lcel.invoke({"review": x["review"],"sentiment": x["sentiment"]}))
    # Step 3: Generate a response based on summary and add it to the input dictionary
    | RunnablePassthrough.assign(response=lambda x: response_chain_lcel.invoke({"review": x["review"],
            "sentiment": x["sentiment"],
            "summary": x["summary"]}))
)


# Test both implementations
def test_chains(review):
    """Test both chain implementations with the given review"""
    print("\n" + "="*50)
    print(f"TESTING WITH REVIEW:\n{review[:100]}...\n")
    
    print("TRADITIONAL CHAIN RESULTS:")
    print(overall_chain.invoke(input={'review':{review}}))
    
    print("\nLCEL CHAIN RESULTS:")
    print(overall_chain_lcel.invoke(input={'review':{review}}))
    
    print("="*50)

# Run tests
test_chains(positive_review)
test_chains(negative_review)

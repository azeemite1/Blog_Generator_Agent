import streamlit as st
from typing import Annotated
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Define the state structure
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    title: str  # Holds the blog title
    content: str  # Holds the blog content

# Initialize Groq Model (Llama3-70B-8192)
model = ChatGroq(model="llama3-70b-8192", temperature=0.7)

def generate_title(state: State):
    """Generates a blog title based on user input."""
    user_idea = state["messages"][-1].content  # Extract user input
    prompt = f"Generate an engaging blog title for this idea: {user_idea}"
    
    title_response = model.invoke([HumanMessage(content=prompt)])
    title = title_response.content.strip()
    
    return {"title": title, "messages": state["messages"] + [AIMessage(content=title)]}

def generate_content(state: State):
    """Generates blog content based on the generated title."""
    title = state["title"]
    prompt = f"Write a detailed and engaging blog post based on this title: {title}"
    
    content_response = model.invoke([HumanMessage(content=prompt)])
    content = content_response.content.strip()
    
    return {"content": content, "messages": state["messages"] + [AIMessage(content=content)]}

def feedback_loop(state: State):
    """Handles feedback from the user to change title, content, or both."""
    # Check if user wants to change title or content
    user_feedback = state["messages"][-1].content.lower()

    if "change title" in user_feedback:
        return "title_generator"
    elif "change content" in user_feedback:
        return "content_generator"
    elif "change both" in user_feedback:
        return "title_generator"  # Re-generate both, starting with the title
    else:
        return END  # No changes requested, end the flow

def make_blog_generator():
    """Creates the blog generator workflow."""
    graph_workflow = StateGraph(State)

    # Add nodes for title and content generation
    graph_workflow.add_node("title_generator", generate_title)
    graph_workflow.add_node("content_generator", generate_content)

    # Define workflow connections
    graph_workflow.add_edge(START, "title_generator")
    graph_workflow.add_edge("title_generator", "content_generator")
    graph_workflow.add_edge("content_generator", END)

    # Add feedback loop for user modifications
    graph_workflow.add_conditional_edges("content_generator", feedback_loop)

    # Compile the workflow
    agent = graph_workflow.compile()
    return agent

# Initialize blog generator agent
agent = make_blog_generator()

# Streamlit Interface
st.title("AI Blog Generator")

# Input for user idea
user_idea = st.text_input("Enter your blog idea:", "")

if user_idea:
    # Generate blog title
    st.write("Generating blog title...")
    result = agent.invoke({"messages": [HumanMessage(content=user_idea)]})
    
    # Display the generated title and content
    blog_title = result["messages"][-1].content
    st.subheader("Generated Blog Title:")
    st.write(blog_title)

    # Generate blog content based on title
    st.write("Generating blog content...")
    result = agent.invoke({"messages": [{"content": blog_title}]})

    blog_content = result["messages"][-1].content
    st.subheader("Generated Blog Content:")
    st.write(blog_content)

    # Ask user for feedback
    user_feedback = st.text_input("Do you want to change the title, content, or both? (e.g., 'change title')", "")

    if user_feedback:
        result = agent.invoke({"messages": [{"content": user_feedback}]})
        st.write("Updated blog:")
        updated_title = result["messages"][-1].content
        st.subheader("Updated Blog Title:")
        st.write(updated_title)
        # Optionally, you can regenerate content as well based on feedback
        if "content" in user_feedback.lower():
            updated_content = result["messages"][-1].content
            st.subheader("Updated Blog Content:")
            st.write(updated_content)
else:
    st.write("Please enter a blog idea to get started.")

# Footer section with developer name
st.markdown("Developed by [Azeem Adeyemi](https://www.linkedin.com/in/azeem-adeyemi-b99291b3/)")


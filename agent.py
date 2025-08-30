import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

# Required imports from LangChain
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Required imports from LangGraph
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables from .env
load_dotenv()

# Retrieve API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Define the agent's graph state ---
class AgentState(TypedDict):
    """
    State structure for the intelligent agent graph.

    Attributes:
        original_transcript (str): The full original transcript of the audio file.
        messages (list): List of messages exchanged during the conversation that are automatically added.
    """
    original_transcript: str
    messages: Annotated[list, add_messages]

class LangGraphAgent:
    """
    Intelligent agent class that manages conversation analysis logic using LangGraph.
    """
    def __init__(self, api_key: str):
        """
        Initialize the intelligent agent with the Gemini model and optimized settings.

        Args:
            api_key (str): API key for Google Generative AI models.
        """
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file. Please set it.")

        # Define the LLM model with optimized settings for analysis and creativity
        self.model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.7  # Higher temperature for more creative responses
        )

        # Initialize memory system to store conversation state
        self.memory = MemorySaver()

        # Build and compile the logical graph
        self.graph = self._build_graph()

    def _build_graph(self):
        """
        Construct the state machine graph using LangGraph.
        """
        graph_builder = StateGraph(AgentState)

        # Define main node: This node calls the LLM to respond to the user
        graph_builder.add_node("chatbot", self.call_chatbot)

        # The graph has a simple loop: always calls the chatbot node
        graph_builder.set_entry_point("chatbot")
        graph_builder.set_finish_point("chatbot")

        # Compile the graph with memory (checkpointing)
        return graph_builder.compile(checkpointer=self.memory)

    def call_chatbot(self, state: AgentState):
        """
        Core function of the agent. Calls the LLM with conversation history and the original transcript.
        """
        system_prompt = (
            "You are an intelligent and specialized language analysis assistant with deep comprehension and advanced reasoning capabilities. "
            "Your task is to analyze and interpret a transcribed conversation, podcast, or speech, and respond to the user's questions in a thoughtful, insightful, and human-like manner. "
            "The user has provided a transcript and expects you to extract viewpoints, key topics, and meaningful insights—not just repeat the content. "
            
            "Core capabilities include: "
            "1. Accurate summarization of the full conversation or specific speaker's contributions. "
            "2. Thematic analysis and identification of key topics discussed. "
            "3. Analytical responses to detailed questions (e.g., What was Person X’s opinion on Topic Y?). "
            "4. Structuring content through mind maps, categorized concepts, or concise bullet points. "
            "5. Communicating naturally and conversationally while maintaining clarity and professionalism. "
            
            "Important instructions: "
            "- Do not copy or paraphrase the transcript unless a direct quote is requested. "
            "- All responses must be based solely on analysis and interpretation of the provided transcript. "
            "- Do not invent or add information not present in the transcript. "
            "- Prioritize clarity, relevance, and alignment with the conversation’s purpose. "
            
            f"Below is the transcript, which is the sole source of truth: {state['original_transcript']} "
            
            "Now, based on conversation history and the user’s current question, provide a clear, thoughtful, and accurate response."
        )

        messages_with_system_prompt = [SystemMessage(content=system_prompt)] + state["messages"]
        response = self.model.invoke(messages_with_system_prompt)
        return {"messages": [response]}

    def get_graph(self):
        """
        Return the compiled graph for use in the main application.
        """
        return self.graph

# --- Create a singleton instance of the agent for Telegram bot usage ---
langgraph_agent_instance = LangGraphAgent(api_key=GEMINI_API_KEY)

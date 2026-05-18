"""
TODO:
1. Incorporate FunctionAgents
2. Remove inheritence to decouple memory and LLM instantiation. 
3. Update the memory parameter to the latest abstractions. 
"""

from src.agents.composable_memory import ComposableMemory
from src.agents.llm_embedding import LLM
from typing import (
    Optional,
    List,
    Literal
)
from llama_index.core.memory import BaseMemory
from llama_index.core.tools import BaseTool
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.agent.openai import OpenAIAgent
from dotenv import load_dotenv

load_dotenv()

class Agent(ComposableMemory, LLM):
    """Agent of your choice with auto setup composable memory.

    Setup the agent of your choice with automatic composable memory and provide a list of tools (carefully read the tools docs as some tools need to be unpacked (*)) along with the other other various parameters already configured. 
    
    """
    def __init__(self):
        pass

    @staticmethod
    def agent(
        agent: Literal["React", "OpenAI"] = "React",
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[BaseMemory] = ComposableMemory.composable_memory() or None,
        llm: LLM = LLM().llm() or Settings.llm,
        verbose: bool = True,
        **kwargs
    ) -> ReActAgent | OpenAIAgent:
        """Agent creation

        Configure the agent of your choice and setup.

        Args:
            agent (str): The agent you would like to setup. Option are: [React, OpenAI]. Defaults to React. 
            tools (List[BaseTool]): An optional list of the tools that your agent can access. Defaults to None.
            memory (BaseMemory): An optional memory component for managing message chains and persisted memory. Defaults to a composable memory object, containing both an primary and secondary layer of abstraction. 
            llm (LLM): The large language model of choice. Defaults to the default configuration within the llm_embeddings module or whichever model was set within llama-index's settings. 
            verbose (bool): Terminal level outputs of the various actions and thoughts as the agent sets out to answer the query in question. Defaults to True. For production settings set this to False. 
            **kwargs: Keyword arguments for further configuration if needed. 

        Returns:
            ReActAgent | OpenAIAgent: An agent object.

        Raises:
            ValueError: If the provided agent is not specified raise error. 
        
        """
        agent_mapper = {
            "React": ReActAgent.from_tools(
                tools=tools,
                llm=llm,
                memory=memory,
                verbose=verbose,
                **kwargs
            ),
            "OpenAI": OpenAIAgent.from_tools(
                tools=tools,
                llm=llm,
                memory=memory,
                verbose=verbose,
                **kwargs
            )
        }

        try:
            return agent_mapper.get(agent)
        except ValueError as e:
            raise ValueError("Provided agent is not supported.")

        
"""
TODO:
1. Update to lastest Llama index memory abstractions!
"""

from typing import (
    Optional,
    List
)
from llama_index.core.prompts import ChatMessage
from llama_index.core.memory import (
    ChatMemoryBuffer,
    VectorMemory,
    SimpleComposableMemory
)
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from dotenv import load_dotenv

load_dotenv()


class ComposableMemory:
    """Composable memory.

    Class for creating and controlling the composable memory for use by an agent.  

    """
    def __init__(self):
        pass
    
    @staticmethod
    def primary_memory_setup(
            chat_history: Optional[List[ChatMessage]] = None,
            token_limit: int = 16000,
            **kwargs
    ) -> ChatMemoryBuffer:
        """Composable memory primary layer setup 

        This method constructs an empty primary abstraction for the composable memory layer. This memory object relates to the ChatMemoryBuffer within Llama index. For more information see - https://docs.llamaindex.ai/en/stable/api_reference/memory/chat_memory_buffer/.

        Args:
            chat_history (list): An optional list of ChatMessages. Defaults to None if not provided. 
            token_limit (int): The limit for the amount of tokens to be stored in the primary abstraction. Defaulted to 16000 tokens.
            kwargs: Additional key word arguments. For more information see - https://docs.llamaindex.ai/en/stable/api_reference/memory/chat_memory_buffer/#llama_index.core.memory.chat_memory_buffer.ChatMemoryBuffer.from_defaults. 

        Returns:
            ChatMemoryBuffer: An memory object which can be controlled independently. 

        """
        return ChatMemoryBuffer.from_defaults(
            chat_history=chat_history,
            token_limit=token_limit,
            **kwargs
        )
    
    @staticmethod
    def secondary_memory_setup(
            vector_store: Optional[BasePydanticVectorStore] = None,
            embed_model: OpenAIEmbedding = Settings.embed_model,
            retriever_kwargs: dict = {"similarity_top_k": 2},
            **kwargs
    ) -> VectorMemory:
        """Composable memory secondary layer setup 

        This method constructs an secondary abstraction for the composable memory layer. This memory object related to the VectorMemory object within Llama index. For more information see - https://docs.llamaindex.ai/en/stable/api_reference/memory/vector_memory/.

        Args:
            vector_store (BasePydanticVectorStore): Vector store of chat messages, defaults to None/empty. NOTE: delete_nodes must be implemented. At time of writing (May 2024), Chroma, Qdrant and SimpleVectorStore all support delete_nodes. For more information see - https://docs.llamaindex.ai/en/stable/api_reference/memory/vector_memory/#llama_index.core.memory.vector_memory.VectorMemory.from_defaults
            embed_model (OpenAIEmbedding): Embedding model for the retrieval of the chat messages. Defaults to Settings embedding model. 
            retriever_kwargs (dict): Optional keyword arguments for retriever usages. Defaults to {"similarity_top_k": 2}.
            kwargs: Any additional keyword arguments that one would like to set. 
        
        """
        return VectorMemory.from_defaults(
            vector_store=vector_store,
            embed_model=embed_model,
            retriever_kwargs=retriever_kwargs,
            **kwargs
        )
    
    @staticmethod
    def composable_memory(
            primary_memory: ChatMemoryBuffer = primary_memory_setup(),
            secondary_memory: VectorMemory = secondary_memory_setup(),
            **kwargs
    ) -> SimpleComposableMemory:
        """Consolidated composable memory setup 

        This method consolidates the composable memory and combines the primary and secondary abstractions into its final object. 

        Args:
            primary_memory (ChatMemoryBuffer): Memory buffer for primary extraction - uses method primary_memory_setup as default where additional tweaks can be configured. 
            secondary_memory (VectorMemory): Memory object for secondary extraction - uses method secondary_memory_setup as default where additional tweaks can be configured.
        
        Returns:
            SimpleComposableMemory
        
        """

        return SimpleComposableMemory.from_defaults(
            primary_memory=primary_memory,
            secondary_memory_sources=[secondary_memory],
            **kwargs
        )
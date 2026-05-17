from dotenv import load_dotenv
from typing import (
    Any
)
import os
from llama_index.core import Settings

load_dotenv()

class LLM:
    """OpenAI LLM of your choice

    Instantiate the OpenAI LLM of your choice with the various parameters to tweak.
    
    """

    def __init__(
        self,
        key: str = os.getenv('OPENAI_API_KEY'),
        model: str = "gpt-4o-mini",
        temp: float = 0.3,
        max_output_window: int = 16384,
        embedding_model: str = "text-embedding-3-large"
    ) -> None:
        """Instantiate the OpenAI model & embedding of your choice.

        Create an instance of the OpenAI agent of your choice, given the most used parameters for LLM configuration. 

        Args:
            key (str): OpenAI API key, defaults to the the API_KEY provided in your environment variables. 
            model (str): The model of your choice, default to gpt-4o-mini.
            temp (float): The LLM's temperature, defaults to 0.3 (deterministic). Values can range between 0 - 1.
            max_output_window (int): The maximum amount of tokens that the model can output. Defaults to 16384 - for more information see https://platform.openai.com/docs/models.
            embedding_model (str): The embedding model for relatedness between pieces of text. Defaults to "text-embedding-3-large" - https://platform.openai.com/docs/models#embeddings.
            **kwargs: Additional keyword arguments for the OpenAI module. For more information see - https://docs.llamaindex.ai/en/stable/api_reference/llms/openai/.
        """
        self._api_key = key 
        self._llm_model = model 
        self._llm_temperature = temp 
        self._output_tokens = max_output_window
        self._embeds_model = embedding_model

    @property
    def api_key(self):
        """Property decorator to manage access to instance variable, with encapsulation and validation.
        
        This method along with the below encapsulates the sensitive instance variable and adds additional validation to raise and error if no API key is provided - NOTE: key should be set as OPENAI_APIKEY as an environment variable.
        """
        return self._api_key
    
    @api_key.setter
    def api_key(self, value):
        if not value:
            raise ValueError('API key cannot be empty - ensure you set up your environment variables correctly.')
        self._api_key = value

    def llm(
        self, 
        host: str = 'OpenAI',
        **kwargs: Any
        ):

        """Large Language Model

        This method updates the settings of the Llama index LLM and returns the instantiated LLM 

        Args:
            host (str): The deployment/host for the LLM. Current options are: [OpenAI, AzureOpenAI]. Defaults to OpenAI.  

        Returns:
            LLM: The large language model as configured. 

        Raises:
            TODO    

        """
        if host == 'OpenAI':
            from llama_index.llms.openai import OpenAI
            llm = Settings.llm = OpenAI(
                model=self._llm_model,
                temperature=self._llm_temperature,
                max_tokens=self._output_tokens,
                api_key=self.api_key,
                **kwargs
            )
        if host == 'AzureOpenAI':
            from llama_index.llms.azure_openai import AzureOpenAI
            llm = Settings.llm = AzureOpenAI(
                model=self._llm_model,
                temperature=self._llm_temperature,
                max_tokens=self._output_tokens,
                api_key=self.api_key,
                **kwargs
            )

        return llm 
    
    def embed_model(
        self,
        host: str = 'OpenAI',
        **kwargs: Any):
        """Embedding model 

        This method updates the settings of the Llama Index embedding model and returns the instantiated embed model. 

        Args:
            host (str): The deployment/host for the LLM. Current options are: [OpenAI, AzureOpenAI]. Defaults to OpenAI.

        Returns:
            Embedding Model: The embedding model as configured. 

        Raises:
            TODO 
        
        """
        if host == 'OpenAI':
            from llama_index.embeddings.openai import OpenAIEmbedding
            embed_model = Settings.embed_model = OpenAIEmbedding(
                model=self._embeds_model,
                api_key=self.api_key,
                **kwargs
            )
        if host == 'AzureOpenAI':
            from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
            embed_model = Settings.embed_model = AzureOpenAIEmbedding(
                model=self._embeds_model,
                api_key=self.api_key,
                **kwargs
            )
        return embed_model
    

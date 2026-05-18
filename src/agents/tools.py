"""
TODO:
1. Introduce a retriever standalone tool. 
2. Wrap router_query_engine as an actual tool with appropriate metadata for LLM discovery. 
3. Unpack embedded tools in all List[FunctionTool] static methods. 
4. Create actual consolidated tool list for ScrapeGraph.
"""

from llama_index.core.tools import (
    QueryEngineTool, 
    ToolMetadata,
    FunctionTool
)
from llama_index.core.tools.tool_spec.load_and_search import LoadAndSearchToolSpec
from llama_index.core.query_engine import CustomQueryEngine, RouterQueryEngine
from llama_index.tools.notion import NotionToolSpec
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from llama_index.tools.weather import OpenWeatherMapToolSpec
from llama_index.tools.google import GoogleSearchToolSpec
from llama_index.tools.requests import RequestsToolSpec
from llama_index.tools.scrapegraph import ScrapegraphToolSpec
from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    TreeIndex,
    KeywordTableIndex,
    PropertyGraphIndex
)
from llama_index.core import Settings
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.tools import QueryEngineTool
from llama_index.core.selectors import BaseSelector
from llama_index.core.schema import NodeWithScore
from llama_index.core.llms.llm import LLM

from typing import (
    Callable, 
    Any,
    Optional,
    List,
    Dict,
    Sequence,
    Literal
)
import os 
from dotenv import load_dotenv

load_dotenv()

class Tools:
    """Create tools for usage by agents directly from the various indexes and other connections setup.

    A whole host of available tools can be seen here and can be incorporated into the module upon requirement - https://llamahub.ai/?tab=tools
    """

    def __init__(self):
        pass 

    @staticmethod
    def query_engine(
        index: VectorStoreIndex | SummaryIndex | TreeIndex | KeywordTableIndex | PropertyGraphIndex,
        name: Optional[str],
        description: Optional[str],
        post_processor: Optional[List[List[NodeWithScore]]],
        **kwargs
    ) -> QueryEngineTool:
        """Method to create a QueryEngineTool
        
        A QueryEngine tool from an index structure provided.

        Args:
            name (str): The tool name - NOTE provide a descriptive name as this is the agents first check when deciding what tool to utilize. 
            description (str): A longer form description of the tool - NOTE there are certain limitations on character lengths depending on the LLM model utilized.
            index: The index structure set up from the ingestion library. Options are [VectorStoreIndex , SummaryIndex , TreeIndex , KeywordTableIndex , PropertyGraphIndex] NOTE you can pass any keyword arguments into the method for example when working with a vector store index you can pass: similarity_top_k = 3.
        
        Returns:
            QueryEngineTool - for more information see https://docs.llamaindex.ai/en/stable/api_reference/tools/query_engine/.
        """
        try:
            qe_tool = QueryEngineTool.from_defaults(
            query_engine=index.as_query_engine(
                llm=Settings.llm,
                node_postprocessors=post_processor
            ),
            name=name,
            description=description,
            **kwargs
        )
            return qe_tool 
        except ValueError:
            raise ValueError
        except KeyError:
            raise KeyError
        
    
    # @staticmethod
    # def router_query_engine(
    #     routing_selector: BaseSelector,
    #     query_engine_tools: Sequence[QueryEngineTool],
    #     # metadata: ToolMetadata,
    #     summarizer: Optional[TreeSummarize] = None,
    #     verbose: bool = False
    # ) -> RouterQueryEngine:
    #     """Method to create a router query engine 

    #     A Router Query engine that take in a user query and a set of "choices" (defined by metadata), and returns one or more selected choices.

    #     Args:
    #         routing_selector (BaseSelector): The selector method to route to the desired node. Refer to [routing](src/routing.py) for the creation of this object. 
    #         query_engine_tools Sequence[QueryEngineTool]: A list of query engine tools over data abstraction of your choice. Refer to the query_engine method within this package to construct the list. 
    #         llm Optional(Llm): An optional LLM object for the engine. Defaults to None. 
    #         summarizer Optional(TreeSummarize): Tree summarizer to summarize sub-results. Default to None. 
    #         verbose (bool): Console log of engine operations. Defaults to False. Should be set to False in production settings. 
        
    #     """
    #     try:
    #         rqe_tool = 
    #         return RouterQueryEngine(
    #             selector=routing_selector,
    #             query_engine_tools=query_engine_tools,
    #             llm=Settings.llm,
    #             summarizer=summarizer,
    #             verbose=verbose
    #         )

    #     except ValueError:
    #         raise ValueError
    #     except KeyError:
    #         raise KeyError
    
    @staticmethod
    def function_tool(
        name: str,
        description: str,
        function:  Optional[Callable[..., Any]] = None
    ) -> FunctionTool:
        """Method to create a FunctionTool 

        A FunctionTool from a well defined (DocString highly advisable) function. NOTE - always ensure that there is a unwanted return value that the agent is aware if something went wrong while calling this function!

        Args:
            name (str): The tool name - NOTE provide a descriptive name as this is the agents first check when deciding what tool to utilize. 
            description (str): A longer form description of the tool - NOTE there are certain limitations on character lengths depending on the LLM model utilized.
            function (Callable): The function name for the LLM to call. 

        Returns:
            FunctionTool - for more information see https://docs.llamaindex.ai/en/stable/api_reference/tools/function/.
        """
        try:
            func_tool = FunctionTool.from_defaults(
                fn=function,
                name=name,
                description=description
            )
        except ValueError:
            raise ValueError
        except KeyError:
            raise KeyError

    
    @staticmethod
    def notion(
        notion_key: str = os.getenv("NOTION_API_KEY"),
        name: str = "Notion",
        description: str = "Find notes or content in Notion."
    ) -> List[FunctionTool]:
        """Create Notion tool 
        
        A Notion tool and automatically automatically sets up a function calling RAG where you can ask questions and append elements to pages. For more information see - https://llamahub.ai/l/tools/llama-index-tools-notion?from=tools. 

        Args:
            notion_key (str): Notion secret key for integration setup. Defaults to environment variable NOTION_API_KEY. For more information see - https://developers.notion.com/docs/create-a-notion-integration.

        Returns:
            List[FunctionTool]: A list of tools that can be utilized by the agent - NOTE - unpack tools if other tools are used within agent. 
        """
        try:
            spec = NotionToolSpec(
                integration_token=notion_key
            )
            tools = LoadAndSearchToolSpec.from_defaults(
                name=name,
                description=description,
                tool=spec.to_tool_list()[0]
            )
            return tools
        except ValueError:
            raise ValueError
        except KeyError:
            raise KeyError
    
    @staticmethod
    def weather(
        open_weather_map_api_key = os.getenv("OPEN_WEATHER_MAP_API_KEY"),
        temp_unit: Literal["celsius","fahrenheit"] = "celsius",
        name: str = "Weather",
        description: str = "This tool can provide the today's and tomorrows weather conditions"
    ) -> List[FunctionTool]:
        """Weather tool

        A weather tool to get the current and future weather (support for current_weather & weather_tomorrow). For more information see https://llamahub.ai/l/tools/llama-index-tools-weather?from=tools.

        Args:
            - open_weather_map_api_key Required(str): Open weather maps API key. Defaults to environment variable OPEN_WEATHER_MAP_API_KEY. For more information see https://openweathermap.org/api.
            - temp_unit Optional(str): The temperature unit of choice. Defaults to celsius.
        Returns 
            List[FunctionTool]: A list of tools that can be utilized by the agent - NOTE - unpack tools if other tools are used within agent. 
        
        """
        try:
            spec = OpenWeatherMapToolSpec(
                key=open_weather_map_api_key,
                temp_units=temp_unit
            )
            tool = LoadAndSearchToolSpec.from_defaults(
                name=name,
                description=description,
                tool=spec.to_tool_list()[0]
            )
            return tool
        except ValueError:
            raise ValueError
        except KeyError:
            raise KeyError
        
    
    @staticmethod
    def duckduckgo(
    ) -> List[FunctionTool]: 
        """DuckDuckGo web search tool. 

        This tool provides functionality to do a web search on DuckDuck go and retrieve information about the query in question. For more information see - https://llamahub.ai/l/tools/llama-index-tools-duckduckgo?from=tools.

        Args:
            None

        Returns:
            List[FunctionTool]: A list of tools that can be utilized by the agent - NOTE - unpack tools if other tools are used within agent. 
        """
        try:
            tools = DuckDuckGoSearchToolSpec().to_tool_list()
        
            return tools
        except ValueError:
            raise ValueError
        except KeyError:
            raise KeyError
    
    @staticmethod
    def googlesearch(
        api_key: str = os.getenv("GOOGLE_API_KEY"),
        engine: str = os.getenv("GOOGLE_ENGINE"),
        name: str = "GoogleSearch",
        description: str = "This tool performs google searches to retrieve information from the internet.",
        **kwargs
    ) -> List[FunctionTool]:
        """Google Search tool.

        This tool provides functionality to do a web search using google - optionally you can set up custom engines that has predefined whitelisted websites for higher level of trust. For more information see - https://llamahub.ai/l/tools/llama-index-tools-google?from=tools.

        Args:
            api_key (str): Your GCP key - https://console.cloud.google.com/ -> API"s & Services -> Credentials -> Create Credentials -> API key. 
            engine (str): The engine"s name. For setup see information -> https://programmablesearchengine.google.com/controlpanel/all

        Returns:
            List[FunctionTool]: A list of tools that can be utilized by the agent - NOTE - unpack tools if other tools are used within agent. 
        """
        try:
            spec = GoogleSearchToolSpec(
                key=api_key,
                engine=engine
            )
            tool = LoadAndSearchToolSpec.from_defaults(
                name=name,
                description=description,
                tool=spec.to_tool_list()[0],
                **kwargs
            ).to_tool_list()
            return tool
        except ValueError:
            raise ValueError
        except KeyError:
            raise KeyError

    @staticmethod
    def httprequest(
        header = dict
    ) -> RequestsToolSpec:
        """Request tool 

        This tool provides functionality to perform HTTP requests. For more information see - https://try:hub.ai/l/tools/llama-index-tools-requests?from=tools.

        Args:
            header (dict): For security reasons, you must specify the hostname for the headers that you wish to provide. As a minimum the header should contain the `Authorization` token and `Content Type`.

        Returns:
            List[FunctionTool]:  A list of tools that can be utilized by the agent - NOTE - unpack tools if other tools are used within agent. 
        
        """
        try:
            spec = RequestsToolSpec(domain_headers=header)
            tool = LoadAndSearchToolSpec.from_defaults(
                name="HTTPRequest",
                description="This tool can perform any HTTP requests.",
                tool=spec.to_tool_list()[0]
            )
            return tool
        except ValueError:
            raise ValueError
        except KeyError:
            raise KeyError
    
    @staticmethod
    def scrapegraph(
        prompt: str,
        url: str,
        api_key: str = os.getenv("SCRAPEGRAPH"),
        return_type: Literal["smart", "markdown", "local"] = "smart",
        schemas: list | dict = None,
        raw: str = None
    ) -> List[Dict]:
        """Web Scraping tool 

        Automatically scrape websites extracting the relevant information you would like to extract. Perfectly paired with structured outputs for quick and efficient information ETL. For more information see - https://llamahub.ai/l/tools/llama-index-tools-scrapegraphai?from=all.

        Args:
            prompt (str): The instruction set given to the agent, i.e. what information you would like to scrape. 
            url (str): The web page url you would like to scrape the information from. 
            api_key (str): The ScrapeGraph API key. Defaults to the environment variable SCRAPEGRAPH. To setup your api key see - https://dashboard.scrapegraphai.com/.
            return_type (str): The method you would like to return the result of the scape as. Option are [smart, markdown, local]. Defaults to smart 
            schemas (list or dict): The structured output schema of choice. Formatted as a list of a pydantic class. Defaults to None - only used with smart return_types. 
            raw (str): Raw text if you would like to perform the return_type = local. Will extract structured data from raw text. Defaults to None - needs to be a valid str object if used with return_type = local. 
        
        Returns:
            FunctionTool: Generic function calling tool for usage by an agent. 
        """
        graph = ScrapegraphToolSpec()


        DEFAULT_PARAMETERS = {
            "smart": {
                "prompt": prompt,
                "url": url,
                "api_key": api_key,
                "schema": schemas
            },
            "markdown": {
                "url": url,
                "api_key": api_key
            },
            "local":{
                "text": raw,
                "api_key": api_key
            }
        }

        match return_type:
            case "smart": response = graph.scrapegraph_smartscraper(DEFAULT_PARAMETERS.get("smart"))
            case "markdown": response = graph.scrapegraph_markdownify(DEFAULT_PARAMETERS.get("markdown"))
            case "local": response = graph.scrapegraph_local_scrape(DEFAULT_PARAMETERS.get("local"))

        return response 

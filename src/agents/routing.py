"""
Routers are modules that take in a user query and a set of "choices" (defined by metadata), and returns one or more selected choices.

They can be used on their own (as "selector modules"), or used as a query engine or retriever (e.g. on top of other query engines/retrievers).

They are simple but powerful modules that use LLMs for decision making capabilities. They can be used for the following use cases and more:

- Selecting the right data source among a diverse range of data sources
- Deciding whether to do summarization (e.g. using summary index query engine) or semantic search (e.g. using vector index query engine)
- Deciding whether to "try" out a bunch of choices at once and combine the results (using multi-routing capabilities).

For more information see - https://docs.llamaindex.ai/en/stable/module_guides/querying/router/

NOTE - there are some interesting router query engines that is not included yet. 
"""
from llama_index.core.selectors import (
    LLMMultiSelector,
    LLMSingleSelector,
    PydanticMultiSelector,
    PydanticSingleSelector
)
from llama_index.core import Settings

from typing import Dict, Any, Literal, Optional

class Routing:

    def __init__(self):
        pass

    @staticmethod
    def selector(
        selector_params: Optional[Dict[str, Any]],
        selector_type: Literal["pysingle", "pymulti", "llmsingle", "llmmulti"] = "pysingle"        
    ) -> LLMMultiSelector | LLMSingleSelector| PydanticMultiSelector | PydanticSingleSelector:

        router_mapping = {
            "pysingle": PydanticSingleSelector,
            "pymulti": PydanticMultiSelector,
            "llmsingle": LLMSingleSelector, 
            "llmmulti": LLMMultiSelector
        }

        DEFAULT_PARAMETERS = {
            "pysingle": {
                "program": None,
                "llm": Settings.llm,
                "prompt_template_str": (
                    "Some choices are given below. It is provided in a numbered list "
                    "(1 to {num_choices}), "
                    "where each item in the list corresponds to a summary.\n"
                    "---------------------\n"
                    "{context_list}"
                    "\n---------------------\n"
                    "Using only the choices above and not prior knowledge, generate "
                    "the selection object and reason that is most relevant to the "
                    "question: '{query_str}'\n"
                ),
                "verbose": False
            },
            "pymulti": {
                "program": None,
                "llm": Settings.llm,
                "prompt_template_str": (
                    "Some choices are given below. It is provided in a numbered "
                    "list (1 to {num_choices}), "
                    "where each item in the list corresponds to a summary.\n"
                    "---------------------\n"
                    "{context_list}"
                    "\n---------------------\n"
                    "Using only the choices above and not prior knowledge, return the top choice(s) "
                    "(no more than {max_outputs}, but only select what is needed) by generating "
                    "the selection object and reasons that are most relevant to the "
                    "question: '{query_str}'\n"
                ),
                "verbose": False
            },
            "llmsingle": {
                "llm": Settings.llm,
                "prompt_template_str": "From the provided user query and the information contained in the metadata return the most relevant node.",
                "output_parser": None
            },
            "llmmulti": {
                "llm": Settings.llm,
                "prompt_template_str": "From the provided user query and the information contained in the metadata return the most relevant node.",
                "output_parser": None,
                "max_outputs": None
            }
        }

        retrieve_router = router_mapping.get(selector_type)

        if router is None:
            raise ValueError("The provided selector_type does not exist.")
        
        router = retrieve_router.from_defaults(selector_params if selector_params else DEFAULT_PARAMETERS.get(selector_type))

        return router

        
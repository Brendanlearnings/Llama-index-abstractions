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

class Routing:

    def __init__(self):
        pass

    @staticmethod
    def selector(
        selector_type: str = "pysingle",
        selector_params: dict = None
    ) -> LLMMultiSelector | LLMSingleSelector| PydanticMultiSelector | PydanticSingleSelector:
        if selector_type == "pysingle":
            try:
                selector = PydanticSingleSelector.from_defaults(
                program=selector_params["pysingle"]["program"] if "program" in selector_params["pysingle"] and selector_params is not None else None,
                llm=Settings.llm,
                prompt_template_str=selector_params["pysingle"]["prompt_template_str"] if "prompt_template_str" in selector_params["pysingle"] and selector_params is not None else (
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
                verbose=selector_params["pysingle"]["verbose"] if "verbose" in selector_params["pysingle"] and selector_params is not None else False
            )
            except ValueError:
                raise ValueError
            except KeyError:
                raise KeyError 
        if selector_type == "pymulti":
            try:
                selector = PydanticMultiSelector.from_defaults(
                    program=selector_params["pymulti"]["program"] if "program" in selector_params["pymulti"] and selector_params is not None else None,
                    llm=Settings.llm,
                    prompt_template_str=selector_params["pymulti"]["prompt_template_str"] if "prompt_template_str" in selector_params["pymulti"] and selector_params is not None else (
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
                    verbose=selector_params["pymulti"]["verbose"] if "verbose" in selector_params["pymulti"] and selector_params is not None else False
                )
            except ValueError:
                raise ValueError
            except KeyError:
                raise KeyError 
        if selector_type == "single":
            try:
                selector = LLMSingleSelector.from_defaults(
                    llm=Settings.llm,
                    prompt_template_str=selector_params["single"]["prompt_template_str"] if "prompt_template_str" in selector_params["single"] and selector_params is not None else "From the provided user query and the information contained in the metadata return the most relevant node.",
                    output_parser=selector_params["single"]["output_parser"] if "output_parser" in selector_params["single"] and selector_params is not None else None
                )
            except ValueError:
                raise ValueError
            except KeyError:
                raise KeyError
        if selector_type == "multi":
            try:
                selector = LLMMultiSelector.from_defaults(
                    llm=Settings.llm,
                    prompt_template_str=selector_params["multi"]["prompt_template_str"] if "prompt_template_str" in selector_params["multi"] and selector_params is not None else "From the provided user query and the information contained in the metadata return the most relevant node.",
                    output_parser=selector_params["multi"]["output_parser"] if "output_parser" in selector_params["multi"] and selector_params is not None else None,
                    max_outputs=selector_params["multi"]["max_outputs"] if "max_outputs" in selector_params["multi"] and selector_params is not None else None
                )
            except ValueError:
                raise ValueError
            except KeyError:
                raise KeyError
            
        return selector
            


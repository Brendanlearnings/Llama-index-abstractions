"""
Node postprocessors are a set of modules that take a set of nodes, and apply some kind of transformation or filtering before returning them.

In LlamaIndex, node postprocessors are most commonly applied within a query engine / retriever, after the node retrieval step and before the response synthesis step.

LlamaIndex offers several node postprocessors for immediate use, while also providing a simple API for adding your own custom postprocessors.

For more information see - https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/. 

TODO:
1. Ensure all packages are up to date and investigate default parameters for each. 
2. Build additional error handling into object validation. 
"""

from typing import List, Optional, Dict, Literal, Required

from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.schema import NodeWithScore
from llama_index.core import Settings
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    KeywordNodePostprocessor,
    MetadataReplacementPostProcessor,
    LongContextReorder,
    SentenceEmbeddingOptimizer,
    SentenceTransformerRerank,
    LLMRerank,
    FixedRecencyPostprocessor,
    EmbeddingRecencyPostprocessor,
    TimeWeightedPostprocessor,
    PIINodePostprocessor,
    PrevNextNodePostprocessor,
    AutoPrevNextNodePostprocessor
)

from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.postprocessor.jinaai_rerank import JinaRerank
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank
from llama_index.postprocessor.colbert_rerank import ColbertRerank

import os 

class PostProcess:

    def __init__(self):
        pass

    @staticmethod
    def post(
        processor_params: Optional[Dict],
        processor: Literal["similarity", "keyword", "metareplace", "longcontext", "sentenceembed", "cohere", "sentencetrans", "llm", "jina", "fixedrecency", "embeddingrecency", "timeweight", "pii", "prevnext", "autoprevnext", "gpt", "colbert"] = "cohere"
    ) -> List[List[NodeWithScore]]:

        post_processor_mapping = {
            "similarity": SimilarityPostprocessor,
            "keyword": KeywordNodePostprocessor,
            "metareplace": MetadataReplacementPostProcessor,
            "longcontext": LongContextReorder,
            "sentenceembed": SentenceEmbeddingOptimizer,
            "cohere": CohereRerank,
            "sentencetrans": SentenceTransformerRerank,
            "llm": LLMRerank,
            "jina": JinaRerank,
            "fixedrecency": FixedRecencyPostprocessor,
            "embeddingrecency": EmbeddingRecencyPostprocessor,
            "timeweight": TimeWeightedPostprocessor,
            "pii": PIINodePostprocessor,
            "prevnext": PrevNextNodePostprocessor,
            "autoprevnext": AutoPrevNextNodePostprocessor,
            "gpt": RankGPTRerank,
            "colbert": ColbertRerank
        }

        DEFAULT_PARAMETERS = {
            "similarity": {
                "similarity_cutoff": 0
            },
            "keyword": {
                "required_keywords": Required[List[str]],
                "exclude_keywords": List[str],
                "lang": "en"
            },
            "metareplace": {
                "target_metadata_key": Required[str]
            },
            "sentenceembed": {
                "embed_model": Settings.embed_model,
                "percentile_cutoff": None,
                "threshold_cutoff": None
            },
            "cohere": {
                "top_n": 2,
                "model": "rerank-english-v2.0",
                "api_key": os.getenv("COHERE_API_KEY"),
                "base_url": None
            },
            "sentencetrans": {
                "top_n": 2,
                "model": "cross-encoder/stsb-distilroberta",
                "device": None,
                "keep_retrieval_score": False,
                "trust_remote_code": True
            },
            "llm": {
                "llm": Settings.llm,
                "choice_select_prompt": None,
                "choice_batch_size": 10,
                "format_node_batch_fn": None,
                "parse_choice_select_answer_fn": None,
                "top_n": 10,
            },
            "jina": {
                "top_n": 2,
                "model": "jina-reranker-v1-base-en",
                "base_url": "https://api.jina.ai/v1",
                "api_key": os.getenv("JINA_API_KEY")
            },
            "fixedrecency": {
                "top_k": 1,
                "date_key": "date"
            },
            "embeddingrecency": {
                "embed_model": Settings.embed_model,
                "date_key": "date",
                "similarity_cutoff": (
                    "The current document is provided.\n"
                    "----------------\n"
                    "{context_str}\n"
                    "----------------\n"
                    "Given the document, we wish to find documents that contain \n"
                    "similar context. Note that these documents are older "
                    "than the current document, meaning that certain details may be changed. \n"
                    "However, the high-level context should be similar.\n"
                )
            },
            "timeweight": {
                "time_decay": 0.99,
                "last_accessed_key": "__last_accessed__",
                "time_access_refresh": True,
                "now": None,
                "top_k": 1
            },
            "pii": {
                "llm": Settings.llm,
                "pii_str_tmpl": (
                    "The current context information is provided. \n"
                    "A task is also provided to mask the PII within the context. \n"
                    "Return the text, with all PII masked out, and a mapping of the original PII "
                    "to the masked PII. \n"
                    "Return the output of the task in JSON. \n"
                    "Context:\n"
                    "Hello Zhang Wei, I am John. "
                    "Your AnyCompany Financial Services, "
                    "LLC credit card account 1111-0000-1111-0008 "
                    "has a minimum payment of $24.53 that is due "
                    "by July 31st. Based on your autopay settings, we will withdraw your payment. "
                    "Task: Mask out the PII, replace each PII with a tag, and return the text. Return the mapping in JSON. \n"
                    "Output: \n"
                    "Hello [NAME1], I am [NAME2]. "
                    "Your AnyCompany Financial Services, "
                    "LLC credit card account [CREDIT_CARD_NUMBER] "
                    "has a minimum payment of $24.53 that is due "
                    "by [DATE_TIME]. Based on your autopay settings, we will withdraw your payment. "
                    "Output Mapping:\n"
                    '{{"NAME1": "Zhang Wei", "NAME2": "John", "CREDIT_CARD_NUMBER": "1111-0000-1111-0008", "DATE_TIME": "July 31st"}}\n'
                    "Context:\n{context_str}\n"
                    "Task: {query_str}\n"
                    "Output: \n"
                    ""
                ),
                "pii_node_info_key": "__pii_node_info__"
            },
            "prevnext": {
                "docstore": Required[BaseDocumentStore],
                "num_nodes": 1,
                "mode": "next"
            },
            "autoprevnext": {
                "docstore": Required[BaseDocumentStore],
                "num_nodes": 1,
                "infer_prev_next_tmpl": (
                    "The current context information is provided. \n"
                    "A question is also provided. \n"
                    "You are a retrieval agent deciding whether to search the "
                    "document store for additional prior context or future context. \n"
                    "Given the context and question, return PREVIOUS or NEXT or NONE. \n"
                    "Examples: \n\n"
                    "Context: Describes the author's experience at Y Combinator."
                    "Question: What did the author do after his time at Y Combinator? \n"
                    "Answer: NEXT \n\n"
                    "Context: Describes the author's experience at Y Combinator."
                    "Question: What did the author do before his time at Y Combinator? \n"
                    "Answer: PREVIOUS \n\n"
                    "Context: Describe the author's experience at Y Combinator."
                    "Question: What did the author do at Y Combinator? \n"
                    "Answer: NONE \n\n"
                    "Context: {context_str}\n"
                    "Question: {query_str}\n"
                    "Answer: "
                ),
                "refine_prev_next_tmpl": (
                    "The current context information is provided. \n"
                    "A question is also provided. \n"
                    "An existing answer is also provided.\n"
                    "You are a retrieval agent deciding whether to search the "
                    "document store for additional prior context or future context. \n"
                    "Given the context, question, and previous answer, "
                    "return PREVIOUS or NEXT or NONE.\n"
                    "Examples: \n\n"
                    "Context: {context_msg}\n"
                    "Question: {query_str}\n"
                    "Existing Answer: {existing_answer}\n"
                    "Answer: "
                ),
                "verbose": False,
                "response_mode": ResponseMode.COMPACT
            },
            "gpt": {
                "top_n": 5,
                "llm": Settings.llm,
                "verbose": False,
                "rankgpt_rerank_prompt": None
            },
            "colbert": {
                "top_n": 5,
                "model": "colbert-ir/colbertv2.0",
                "tokenizer": "colbert-ir/colbertv2.0",
                "device": None,
                "keep_retrieval_score": False
            }
        }

        if processor in ["keyword", "metareplace", "prevnext", "autoprevnext"] and processor_params is None:
            raise ValueError(f"The provided post processor has required values that needs to be provided in the processor_params object.")

        retrieve_processor = post_processor_mapping.get(processor)
        if processor is None:
            raise ValueError("Provided processor does not exist in collection.")
        
        post_processor = retrieve_processor.model_validate(processor_params if processor_params else DEFAULT_PARAMETERS.get(processor))

        return post_processor

        
        



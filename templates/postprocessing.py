"""
Node postprocessors are a set of modules that take a set of nodes, and apply some kind of transformation or filtering before returning them.

In LlamaIndex, node postprocessors are most commonly applied within a query engine, after the node retrieval step and before the response synthesis step.

LlamaIndex offers several node postprocessors for immediate use, while also providing a simple API for adding your own custom postprocessors.

For more information see - https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/. 
"""

from typing import List
from llama_index.core.schema import NodeWithScore
from llama_index.core import Settings
import os 

class PostProcess:

    def __init__(self):
        pass

    @staticmethod
    def post(
        processors: list = ["similarity"],
        processor_params: dict = None
    ) -> List[List[NodeWithScore]]:
        processor_list = []
        for proc in processors:
            if proc == "similarity":
                from llama_index.core.postprocessor import SimilarityPostprocessor
                try:
                    processor_list.append(SimilarityPostprocessor(
                        similarity_cutoff=processor_params["similarity"]["similarity_cutoff"] if "similarity_cutoff" in processor_params["similarity"] else 0
                        )
                    )
                except ValueError as e:
                    raise ValueError
                except KeyError as e:
                    raise KeyError
            if proc == "keyword":
                from llama_index.core.postprocessor import KeywordNodePostprocessor
                try:
                    processor_list.append(KeywordNodePostprocessor(
                        required_keywords=processor_params["keyword"]["required_keywords"],
                        exclude_keywords=processor_params["keyword"]["exclude_keywords"],
                        lang=processor_params["keyword"]["exclude_kelangywords"]
                    ))
                except ValueError as e:
                    raise ValueError
                except KeyError as e:
                    raise KeyError
            if proc == "metadata":
                from llama_index.core.postprocessor import MetadataReplacementPostProcessor
                try:
                    processor_list.append(MetadataReplacementPostProcessor(
                        target_metadata_key=processor_params["metadata"]["target_metadata_key"]
                    ))
                except ValueError as e:
                    raise ValueError
                except KeyError as e:
                    raise KeyError
            if proc == "longcontent":
                from llama_index.core.postprocessor import LongContextReorder
                try:
                    processor_list.append(LongContextReorder())
                except ValueError as e:
                    raise ValueError
                except KeyError as e:
                    raise KeyError
            if proc == "sentenceembed":
                from llama_index.core.postprocessor import SentenceEmbeddingOptimizer
                try:
                    processor_list.append(SentenceEmbeddingOptimizer(
                        embed_model=Settings.embed_model,
                        percentile_cutoff=processor_params["sentenceembed"]["percentile_cutoff"] if "percentile_cutoff" in processor_params["sentenceembed"] else None,
                        threshold_cutoff=processor_params["sentenceembed"]["threshold_cutoff"] if "threshold_cutoff" in processor_params["sentenceembed"] else None,
                    ))
                except ValueError as e:
                    raise ValueError
                except KeyError as e:
                    raise KeyError
            if proc == "cohere":
                from llama_index.postprocessor.cohere_rerank import CohereRerank
                try:
                    processor_list.append(CohereRerank(
                        top_n=processor_params["cohere"]["top_n"] if "top_n" in processor_params["cohere"] else 2,
                        model=processor_params["cohere"]["model"] if "model" in processor_params["cohere"] else "rerank-english-v2.0",
                        api_key=os.getenv("COHERE_API_KEY"),
                        base_url=processor_params["cohere"]["base_url"] if "base_url" in processor_params["cohere"] else None
                    ))
                except ValueError as e:
                    raise ValueError
                except KeyError as e:
                    raise KeyError
            if proc == "sentencetransform":
                from llama_index.core.postprocessor import SentenceTransformerRerank
                try:
                    processor_list.append(SentenceTransformerRerank(
                        top_n=processor_params["sentencetransform"]["top_n"] if "top_n" in processor_params["sentencetransform"] else 2,
                        model=processor_params["sentencetransform"]["model"] if "model" in processor_params["sentencetransform"] else "cross-encoder/stsb-distilroberta",
                        device=processor_params["sentencetransform"]["device"] if "device" in processor_params["sentencetransform"] else None,
                        keep_retrieval_score=processor_params["sentencetransform"]["keep_retrieval_score"] if "keep_retrieval_score" in processor_params["sentencetransform"] else False,
                        trust_remote_code=processor_params["sentencetransform"]["trust_remote_code"] if "trust_remote_code" in processor_params["sentencetransform"] else True
                    ))
                except ValueError as e:
                    raise ValueError
                except KeyError as e:
                    raise KeyError    
            if proc == "llm":
                from llama_index.core.postprocessor import LLMRerank
                try:
                    processor_list.append(LLMRerank(
                        llm=Settings.llm,
                        choice_select_prompt=processor_params["llm"]["choice_select_prompt"] if "choice_select_prompt" in processor_params["llm"] else None,
                        choice_batch_size=processor_params["llm"]["choice_batch_size"] if "choice_batch_size" in processor_params["llm"] else 10,
                        format_node_batch_fn=processor_params["llm"]["format_node_batch_fn"] if "format_node_batch_fn" in processor_params["llm"] else None,
                        parse_choice_select_answer_fn=processor_params["llm"]["parse_choice_select_answer_fn"] if "parse_choice_select_answer_fn" in processor_params["llm"] else None,
                        top_n=processor_params["llm"]["top_n"] if "top_n" in processor_params["llm"] else 10
                    ))
                except ValueError as e:
                    raise ValueError
                except KeyError as e:
                    raise KeyError 
            if proc == 'jina':
                from llama_index.postprocessor.jinaai_rerank import JinaRerank
                try:
                    processor_list.append(JinaRerank(
                        top_n=processor_params["jina"]["top_n"] if "top_n" in processor_params["jina"] else 2,
                        model=processor_params["jina"]["model"] if "model" in processor_params["jina"] else "jina-reranker-v1-base-en",
                        base_url=processor_params["jina"]["base_url"] if "base_url" in processor_params["jina"] else "https://api.jina.ai/v1",
                        api_key=os.getenv("JINA_API_KEY")
                    ))
                except ValueError as e:
                    raise ValueError
                except KeyError as e:
                    raise KeyError 
            if proc == 'fixedrecency':
                from llama_index.core.postprocessor import FixedRecencyPostprocessor
                try:
                    processor_list.append(FixedRecencyPostprocessor(
                        top_k=processor_params["fixedrecency"]["top_k"] if "top_k" in processor_params["fixedrecency"] else 1,
                        date_key=processor_params["fixedrecency"]["date_key"] if "date_key" in processor_params["fixedrecency"] else "date"
                    ))
                except ValueError as e:
                    raise ValueError
                except KeyError as e:
                    raise KeyError 
            if proc == "embeddingrecency":
                from llama_index.core.postprocessor import EmbeddingRecencyPostprocessor
                try:
                    processor_list.append(EmbeddingRecencyPostprocessor(
                        embed_model=Settings.embed_model,
                        date_key=processor_params["embeddingrecency"]["date_key"] if "date_key" in processor_params["embeddingrecency"] else "date",
                        similarity_cutoff=processor_params["embeddingrecency"]["similarity_cutoff"] if "similarity_cutoff" in processor_params["embeddingrecency"] else 0.7,
                        query_embedding_tmpl=processor_params["embeddingrecency"]["query_embedding_tmpl"] if "query_embedding_tmpl" in processor_params["embeddingrecency"] else (
    "The current document is provided.\n"
    "----------------\n"
    "{context_str}\n"
    "----------------\n"
    "Given the document, we wish to find documents that contain \n"
    "similar context. Note that these documents are older "
    "than the current document, meaning that certain details may be changed. \n"
    "However, the high-level context should be similar.\n"
)
                    ))
                except ValueError as e:
                    raise ValueError
                except KeyError as e:
                    raise KeyError 
            if proc == "timeweight":
                from llama_index.core.postprocessor import TimeWeightedPostprocessor
                try:
                    processor_list.append(TimeWeightedPostprocessor(
                        time_decay=processor_params["timeweight"]["time_decay"] if "time_decay" in processor_params["timeweight"] else 0.99,
                        last_accessed_key=processor_params["timeweight"]["last_accessed_key"] if "last_accessed_key" in processor_params["timeweight"] else "__last_accessed__",
                        time_access_refresh=processor_params["timeweight"]["time_access_refresh"] if "time_access_refresh" in processor_params["timeweight"] else True,
                        now=processor_params["timeweight"]["now"] if "now" in processor_params["timeweight"] else None,
                        top_k=processor_params["timeweight"]["top_k"] if "top_k" in processor_params["timeweight"] else 1
                    ))
                except ValueError as e:
                    raise ValueError
                except KeyError as e:
                    raise KeyError 
            if proc == "piinode":
                from llama_index.core.postprocessor import PIINodePostprocessor
                try:
                    processor_list.append(PIINodePostprocessor(
                        llm=Settings.llm,
                        pii_str_tmpl=processor_params["piinode"]["pii_str_tmpl"] if "pii_str_tmpl" in processor_params["piinode"] else (
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
                        pii_node_info_key=processor_params["piinode"]["pii_node_info_key"] if "pii_node_info_key" in processor_params["piinode"] else "__pii_node_info__"
                    ))
                except ValueError as e:
                    raise ValueError
                except KeyError as e:
                    raise KeyError    
            if proc == "prevnext":
                from llama_index.core.postprocessor import PrevNextNodePostprocessor
                try:
                    processor_list.append(PrevNextNodePostprocessor(
                        docstore=processor_params["prevnext"]["docstore"],
                        num_nodes=processor_params["prevnext"]["num_nodes"] if "num_nodes" in processor_params["prevnext"] else 1,
                        mode=processor_params["prevnext"]["mode"] if "mode" in processor_params["prevnext"] else "next"
                    ))
                except ValueError as e:
                    raise ValueError
                except KeyError as e:
                    raise KeyError 
            if proc == "autoprevnext":
                from llama_index.core.postprocessor import AutoPrevNextNodePostprocessor
                from llama_index.core.response_synthesizers import ResponseMode
                try:
                    processor_list.append(AutoPrevNextNodePostprocessor(
                        docstore=processor_params["autoprevnext"]["docstore"],
                        llm=Settings.llm,
                        num_nodes=processor_params["autoprevnext"]["num_nodes"] if "num_nodes" in processor_params["autoprevnext"] else 1,
                        infer_prev_next_tmpl=processor_params["autoprevnext"]["infer_prev_next_tmpl"] if "infer_prev_next_tmpl" in processor_params["autoprevnext"] else (
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
                        refine_prev_next_tmpl=processor_params["autoprevnext"]["refine_prev_next_tmpl"] if "refine_prev_next_tmpl" in processor_params["autoprevnext"] else (
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
                        verbose=processor_params["autoprevnext"]["verbose"] if "verbose" in processor_params["autoprevnext"] else False,
                        response_mode=processor_params["autoprevnext"]["response_mode"] if "response_mode" in processor_params["autoprevnext"] else ResponseMode.COMPACT
                    ))
                except ValueError as e:
                    raise ValueError
                except KeyError as e:
                    raise KeyError
            if proc == "gpt":
                from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank
                try:
                    processor_list.append(RankGPTRerank(
                        top_n=processor_params["gpt"]["top_n"] if "top_n" in processor_params["gpt"] else 5,
                        llm=Settings.llm,
                        verbose=processor_params["gpt"]["verbose"] if "verbose" in processor_params["gpt"] else False,
                        rankgpt_rerank_prompt=processor_params["gpt"]["rankgpt_rerank_prompt"] if "rankgpt_rerank_prompt" in processor_params["gpt"] else None         
                    ))
                except ValueError as e:
                    raise ValueError
                except KeyError as e:
                    raise KeyError
            if proc == "colbert":
                from llama_index.postprocessor.colbert_rerank import ColbertRerank
                try:
                    processor_list.append(ColbertRerank(
                        top_n=processor_params["colbert"]["top_n"] if "top_n" in processor_params["colbert"] else 5,
                        model=processor_params["colbert"]["model"] if "model" in processor_params["colbert"] else "colbert-ir/colbertv2.0",
                        tokenizer=processor_params["colbert"]["tokenizer"] if "tokenizer" in processor_params["colbert"] else "colbert-ir/colbertv2.0",
                        device=processor_params["colbert"]["device"] if "device" in processor_params["colbert"] else None,
                        keep_retrieval_score=processor_params["colbert"]["keep_retrieval_score"] if "keep_retrieval_score" in processor_params["colbert"] else False
                    ))
                except ValueError as e:
                    raise ValueError
                except KeyError as e:
                    raise KeyError
            # if proc == "rankllm":
            #     from llama_index.core.



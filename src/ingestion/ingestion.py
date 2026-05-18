"""
TODO:
1. Incorporate further readers into package. 
2. Update Llama Parse / Cloud to V2 - full update of extract file.
3. Enhance Storage Context to integrate better into Weaviate
4. Enhance Error Handling
5. Update docstrings
6. Decouple weaviate lib. 
"""
from dotenv import load_dotenv
import os
from typing import List, Type, Sequence, Literal, Dict, Any, Optional
from pydantic import BaseModel 

from llama_index.core import Settings, Document
from llama_index.core.schema import BaseNode
from llama_index.core import (
    StorageContext, 
    VectorStoreIndex, 
    SummaryIndex,
    TreeIndex,
    KeywordTableIndex,
    PropertyGraphIndex
)
from llama_parse import LlamaParse
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    SummaryIndex,
    TreeIndex,
    KeywordTableIndex,
    PropertyGraphIndex
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import (
        SummaryExtractor,
        QuestionsAnsweredExtractor,
        TitleExtractor,
        KeywordExtractor,
)
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser.text.utils import split_by_sentence_tokenizer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.node_parser.text.utils import split_by_sentence_tokenizer
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.vector_stores.weaviate import WeaviateVectorStore

from src.ingestion.vector_db import VectorDB

load_dotenv()

class Ingestion:
    """
    This class solves for all data ingestion related activities to index creation as a pipeline of methods. Conformed to follow typical Data Engineering principles (ETL) with an additional method to construct the final object.  
    """

    def __init__(self):
        pass

    @staticmethod
    def extract_file(
        input: str,
        multiple: bool = False,
        parse: bool = True,
        parse_output: str = "markdown",
        parse_mode: Literal["auto", "continuous", "fast", "premium"] = "auto",
        parse_file_type: str = ".pdf",
        progress_ind: bool = True,
        **kwargs
    ) -> List[Document] | Document:
        """Extract content from a document/file. 

        This method provides the functionality to extract the contents of supported document types (https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/#supported-file-types) to a Llama Index Document or List[Documents]. For more information regarding the reader see - https://docs.llamaindex.ai/en/stable/api_reference/readers/simple_directory_reader/.
        State of the art parsing is also made available through the LlamaParse package - the main goal of LlamaParse is to parse and clean your data, ensuring that it's good quality before passing to any downstream LLM use case such as advanced RAG. For more information see - https://docs.llamaindex.ai/en/stable/llama_cloud/llama_parse/#llamaparse. NOTE - this is a cloud based solution and should be vetted against FR security policies. 

        Args:
            input (str): The input method you would like utilize. Represented as a path to a file or directory, an S3 bucket URL or URL. This should align to the below mentioned parameters to ensure successful extraction. For more information see - https://docs.cloud.llamaindex.ai/llamaparse/input.
            multiple (bool): Boolean indicator for single or multiple document extraction. If single the input must be the path to your file (directory/file.extension), if multiple this should be the path to the desired directory. Defaults to False. 
            parse (bool): Boolean indicator for parsing the contents of the file using Llama Parse. If set to False the accompanying parse_* parameters can be omitted. Defaults to True. 
            parse_output (str): The output type post parsing you would like to utilize. Defaults to industry standard of markdown (tried and tested). For more information see - https://docs.cloud.llamaindex.ai/llamaparse/output_modes/
            parse_mode (str): The parsing modes you would like to utilize. NOTE - each mode results in different billing metrics, and supports different output formats. The options are ["auto", "continuous", "fast", "premium"]. Defaults to auto. For more information see - https://docs.cloud.llamaindex.ai/llamaparse/output_modes/#parsing-modes-and-output.
            parse_file_type (str): The file extension you are setting out to parse. Llama Parse supports up to 10 file types. 
            progress_ind (bool): Console logs of progress of extraction. Defaults to True. 
            **kwargs: Additional key word arguments that can be provided to the LlamaParse object. For more information see - https://docs.cloud.llamaindex.ai/llamaparse/features/python_usage

        Returns:
            A list of Llama Index object type Documents or a single Document.  
        """

        parse_format = {
            "auto": "auto_mode",
            "continuous": "continuous_mode",
            "fast": "fast_mode",
            "premium": "premium_mode"
        }
        parse_param = parse_format.get(parse_mode)
        parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            parse_param = parse_mode,
            result_type=parse_output,
            **kwargs
        )
        file_extractor = {parse_file_type: parser}
        # Conditional checks to account for variations in parameters settings. Creates the nodes from the input document/s.
        if multiple and parse:
            documents = SimpleDirectoryReader(
                input_dir=input,
                file_extractor=file_extractor            
            ).load_data(show_progress=progress_ind)
            
        if not multiple and parse:
            documents = SimpleDirectoryReader(
                input_files=[input],
                file_extractor=file_extractor
            ).load_data(show_progress=progress_ind)

        if not multiple and not parse:
            documents = SimpleDirectoryReader(
                input_files=[input]
            ).load_data(show_progress=progress_ind)

        if multiple and not parse:
            documents = SimpleDirectoryReader(
                input_dir=input
            ).load_data()

        return documents
    
    @staticmethod
    def transform(
        documents: List[Document] | Document,
        transformation_params: Dict[str, Any],
        metadata_collectors: Optional[List[Literal["title", "summary", "QA", "keyword"]]],
        transformation_type: Literal["semantic", "sentence", "sentence_window", "token"] = ["semantic"],
    ) -> Sequence[BaseNode]:
        """Transformation to be applied over supplied Document/s.

        This method provides you with a base (can be updated to include more) of transformations that can be applied over the document to split the content up into Nodes using a transformation pipeline. 
        By default metadata extractors will be incorporated to decorate your Node, which is useful for down the line routing etc. As well as the set embedding model you are utilizing. 

        Args:
            documents (List[Documents] or Document): The relevant Llama Index object that you would like to apply the transformations over. 
            transformation_type (str): The type of transformation you would like to apply over the document. Options are ["semantic", "sentence", "sentence_window", "token"]. Defaults to token.
            transformation_params (dict): A dict object containing the key and value pair for the given parameter keyword arguments. Auto defaults applied. Defaults to None. 

        Returns:
            Sequence[BaseNode]: A list of base nodes after passing through the transformation pipeline according to the parameter settings along with their relevant metadata elements. TODO - can further investigate more metadata additions. 

        Documentation:
            See the below links to each individual transformation package:
            - Semantic: https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/semantic_splitter/
            - Sentence: https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/sentence_splitter/
            - Sentence Window: https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/sentence_window/#llama_index.core.node_parser.SentenceWindowNodeParser
            - Token Text: https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/token_text_splitter/#llama_index.core.node_parser.TokenTextSplitter
        """
        transformations = []
        if not transformation_params:
            raise ValueError('Kindly specify the transformation you would like to apply.')
        
        splitter_mapping = {
            "semantic": SemanticSplitterNodeParser(
                        buffer_size=transformation_params["parameters"]["buffer_size"] if "buffer_size" in transformation_params["parameters"] else 1,
                        embed_model=OpenAIEmbedding(model=Settings.embed_model, api_key=os.getenv("OPENAI_API_KEY")),
                        sentence_splitter=transformation_params["parameters"]["sentence_splitter"] if "sentence_splitter" in transformation_params["parameters"] else split_by_sentence_tokenizer(),
                        include_metadata=transformation_params["parameters"]["include_metadata"],
                        include_prev_next_rel=transformation_params["parameters"]["include_prev_next_rel"],
                        breakpoint_percentile_threshold=transformation_params["parameters"]["breakpoint_percentile_threshold"] if "breakpoint_percentile_threshold" in transformation_params["parameters"] else 95
            ),
            "sentence": SentenceSplitter(
                        chunk_size=transformation_params["parameters"]["chunk_size"] if "chunk_size" in transformation_params["parameters"] else 1024,
                        chunk_overlap=transformation_params["parameters"]["chunk_overlap"] if "chunk_overlap" in transformation_params["parameters"] else 200,
                        separator=transformation_params["parameters"]["separator"] if "separator" in transformation_params["parameters"] else ' ',
                        paragraph_separator=transformation_params["parameters"]["paragraph_separator"] if "paragraph_separator" in transformation_params["parameters"] else '\n\n\n',
                        secondary_chunking_regex=transformation_params["parameters"]["secondary_chunking_regex"] if "secondary_chunking_regex" in transformation_params["parameters"] else '[^,.;。？！]+[,.;。？！]?'
            ),
            "sentence_window": SentenceWindowNodeParser(
                        sentence_splitter=transformation_params["parameters"]["sentence_splitter"] if "sentence_splitter" in transformation_params["parameters"] else split_by_sentence_tokenizer(), 
                        include_metadata=transformation_params["parameters"]["include_metadata"], 
                        include_prev_next_rel=transformation_params["parameters"]["include_prev_next_rel"], 
                        window_size=transformation_params["parameters"]["window_size"] if "window_size" in transformation_params["parameters"] else 3, 
                        window_metadata_key=transformation_params["parameters"]["window_metadata_key"] if "window_metadata_key" in transformation_params["parameters"] else 'window', 
                        original_text_metadata_key=transformation_params["parameters"]["original_text_metadata_key"] if "original_text_metadata_key" in transformation_params["parameters"] else 'original_text', 
            ),
            "token": TokenTextSplitter(
                        chunk_size=transformation_params["parameters"]["chunk_size"] if "chunk_size" in transformation_params["parameters"] else 1024,
                        chunk_overlap=transformation_params["parameters"]["chunk_overlap"] if "chunk_overlap" in transformation_params["parameters"] else 20
            )
        }

        metadata_extractor_mapping = {
            "title": TitleExtractor,
            "summary": SummaryExtractor,
            "QA": QuestionsAnsweredExtractor,
            "keyword": KeywordExtractor
        }

        try:
            transformations = [splitter for splitter in splitter_mapping.get(transformation_type)].extend([metadata_extractor_mapping.get(metdata_extractor)() for metdata_extractor in metadata_collectors])
        except ValueError as e:
            raise ValueError(f"The provided value in the transformations params caused the following error: {e}")

        
    
        pipeline = IngestionPipeline(
            transformations=transformations
        )
        if isinstance(documents, Document):
            nodes = pipeline.run(
                documents=documents
            )
        elif isinstance(documents, list):
            nodes = pipeline.run(
                documents=[documents]
            )

        return nodes
        
    @staticmethod    
    def load(
        index_name: str,
        nodes: Sequence[BaseNode] = None
    ) -> VectorStoreIndex:
        """Load data to Vector DB (Weaviate)

        This method provides you with the functionality to load your nodes into your vector database and returns an vector store index that can be directly utilized. 

        Args:
            index_name (str): The name of the index/collection you want to create or consume. 
            nodes (Sequence[BaseNode]): Llama Index list of base nodes. This can be omitted if the collection already exists within your vector database. 

        Returns:
            VectorStoreIndex: An vector index over your collection of nodes. Can directly be used as as query engine as part of a tool.
        """
        vector_store = WeaviateVectorStore(
            weaviate_client=VectorDB().client,
            index_name=index_name,
        )
        storage = StorageContext.from_defaults(
            vector_store=vector_store
        )
        return VectorStoreIndex(
            nodes=nodes,
            storage_context=storage,
            embed_model=Settings.embed_model
        )

    @staticmethod
    def consume(
        nodes: Sequence[BaseNode],
        index_type: Literal["vector", "summary", "tree", "keyword", "graph"] = "vector",
        progress_ind: bool = False,
        **kwargs
    ) -> SummaryIndex | TreeIndex | KeywordTableIndex | PropertyGraphIndex:
        """Consume the in memory index of choice. 

        This method provides you with functionality to construct an in memory abstraction of the Llama index Index type of your choice.

        Args:
            nodes (Sequence[BaseNode]): The list of nodes used to create the index abstraction over. 
            index_type (str): The type of index you would like to create over your list of nodes. Options are ['summary', "tree", "keyword", 'graph']. Defaults to  For more information see - https://docs.llamaindex.ai/en/stable/module_guides/indexing/index_guide/
            progress_ind (bool): Boolean indicator to show the progress of constructing the in memory index from your list of nodes. Defaults to True. 

        Returns:
            SummaryIndex | TreeIndex | KeywordTableIndex | PropertyGraphIndex: The in memory index abstraction of choice. 
        
        """
        index_mapping = {
            "vector": VectorStoreIndex,
            "summary": SummaryIndex,
            "tree": TreeIndex,
            "keyword": KeywordTableIndex,
            "graph": PropertyGraphIndex
        }

        index = index_mapping.get(index_type)

        if isinstance(index, VectorStoreIndex):
            vector_with_params = index.model_validate(
                  {
                       "nodes": nodes,
                       "embed_model": embed_model or Settings.embed_model,
                       "show_progress": progress_ind,
                       **kwargs
                  }
             )
            return vector_with_params
        
        index_with_parameters = index.model_validate(
             {
                  "nodes": nodes,
                  "show_progress": progress_ind,
                  **kwargs
             }
        )
                
        return index
        

import weaviate
from llama_index.core import StorageContext
import weaviate.classes as wvc
from weaviate.classes.config import Integrations
from weaviate.exceptions import WeaviateBaseError, WeaviateInvalidInputError, WeaviateQueryError
from weaviate.collections.collection import Collection
from weaviate.classes.init import Auth
import os 
from dotenv import load_dotenv

load_dotenv()

class VectorDB:

    def __init__(
        self,
        type: str = "embedded"
    ):
        """Initialize DB object

        Args:
            type (str): The type of connection you would like to perform. Options are ["local","embedded"]. For more information see https://weaviate.io/developers/weaviate/installation.
        
        """
        if type == "local":
            self.client = weaviate.connect_to_local() # For server hosted instance - on local machine from docker image
        if type == "embedded":
            self.client = weaviate.connect_to_embedded(
                version="1.26.5",  # e.g. version="1.26.5"
                headers={
                    "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")  # Replace with your API key
                },
            )
        if type == "cloud":
            self.client = weaviate.connect_to_weaviate_cloud(
                            cluster_url=os.getenv("WEAVIATE_URL"),
                            auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
                            additional_config=wvc.init.AdditionalConfig(timeout=wvc.init.Timeout(init=10))
                        )
        

    def batch_sizing(
        self,
        collection: str,
        batch_setting: str = 'Dynamic',
        reference: bool = False,
        fixed_size_params: dict = None,
        concurrent_requests_fixed: int = 600,
        **kwargs
        
    ) -> weaviate.types.UUID:
        """Batch sizing configuration

        This method allows you to configure the batch sizing options based on the parameters of your choice. For more information see - https://weaviate.io/developers/weaviate/client-libraries/python#batch-sizing.
        We recommend using the collection object to perform batch imports of single collections or tenants. If you are importing objects across many collections, such as in a multi-tenancy configuration, using client.batch may be more convenient.

        Args:
        collection (str): The name of the DB collection object you are configuring the batch sizing for. 
        batch_setting (str): The method of setting up your batch sizing. The following values are allowed [Dynamic, Fixed, Rate]. Defaults to Dynamic sizing. 
        reference (bool): Boolean option to add a reference object instead of a batch object. NOTE - if set to true all parameters should be passed as keyword arguments! Defaults to False. 
        fixed_size_params (dict): fixed sizing parameters as a dict object representing the keyword argument for the function, eg {"batch_size": 100, "concurrent_requests":4}. Defaults to None.
        concurrent_requests_fixed (int): The amount of concurrent requests per minute for the fixed rate sizing option. Defaults to 600.
        **kwargs: Additional options for the add_object / add_reference function. 

        Returns:
        weaviate.types.UUID: The unique identifier for the batch configuration sizing setup. For further context to failed objects use client.batch.failed_objects / client.batch.failed_references
        
        """
        if not reference:
            if batch_setting == 'Dynamic':
                try:
                    with self.client.batch.dynamic() as batch:
                        batch_size = batch.add_object(
                            collection=collection,
                            **kwargs,
                        )
                except Exception as e:
                    print(f"Dymanic batch sizing + object failed: {e}")

            if batch_setting == 'Fixed':
                try:
                    with self.client.batch.fixed_size(*[fixed_size_params[key] for key in  fixed_size_params.items()]) as batch:
                        batch_size = batch.add_object(
                                collection=collection,
                                **kwargs,
                            )
                except Exception as e:
                    print(f"Fixed batch sizing + object  failed: {e}")

            if batch_setting == 'Rate':
                try:
                    with self.client.batch.rate_limit(requests_per_minute=concurrent_requests_fixed) as batch:
                        batch_size = batch.add_object(
                                collection=collection,
                                **kwargs,
                            )
                except Exception as e:
                    print(f"Rate batch sizing + object  failed: {e}")
        else:
            if batch_setting == 'Dynamic':
                try:
                    with self.client.batch.dynamic() as batch:
                        batch_size = batch.add_reference(
                            **kwargs,
                        )
                except Exception as e:
                    print(f"Dynamic batch sizing + reference failed: {e}")

            if batch_setting == 'Fixed':
                try:
                    with self.client.batch.fixed_size(*[fixed_size_params[key] for key in  fixed_size_params.items()]) as batch:
                        batch_size = batch.add_reference(
                                **kwargs,
                            )
                except Exception as e:
                    print(f"Fixed batch sizing + reference failed: {e}")

            if batch_setting == 'Rate':
                try:
                    with self.client.batch.rate_limit(requests_per_minute=concurrent_requests_fixed) as batch:
                        batch_size = batch.add_reference(
                                **kwargs,
                            )
                except Exception as e:
                    print(f"Rate batch sizing + reference failed: {e}")
        return batch_size

    def collection_manager(
        self,
        name: str,
        # deployment: str = 'OpenAI',
        properties: list = None,
        create: bool = False
    ) -> Collection:
        """Create/Instantiate a collection

        You can instantiate a collection object by creating a collection, or by retrieving an existing collection. For more information see - https://weaviate.io/developers/weaviate/client-libraries/python#working-with-collections.

        Args:
        name (str): The name of the collection in question. 
        deployment (str): The deployment for the infrastructure behind the LLM. Options are ['OpenAI','AzureOpenAI']. Defaults to OpenAI. 
        properties (tuple): The various properties for the given collection. Useful when applying routing and metadata filtering. Contains weaviate instantiated objects. Various submodules available to control this more effectively, but useful if you want to instantiate the collection once off with the various properties available. Defaults to None. 
        create_or_get (bool): Create a new collection or get an existing one. Defaults to creating a new collection. 

        Returns
        collections: Weaviate collections object. 
        
        """
        if create:
            try:
                collection = self.client.collections.create(
                    name=name,
                    # vectorizer_config= wvcc.Configure.Vectorizer.text2vec_openai() if deployment == 'OpenAI' else wvcc.Configure.Vectorizer.text2vec_azure_openai(),
                    # generative_config= wvcc.Configure.Generative.openai() if deployment == 'OpenAI' else wvcc.Configure.Generative.azure_openai(),
                    properties=properties
                )
                return collection
            except Exception as e:
                return None
        else:
            try:
                collection = self.client.collections.get(name=name)
                return collection
            except Exception as e:
                return None

    def collection_data_submod(
        self,
        type: str,
        collection: Collection,
        delete_many_filter: dict = None,
        update_options: str = 'add',
        **kwargs
    ) -> None: 
        """
        
        """
        try:
            if type == "insert":
                object = collection.data.insert(
                    **kwargs
                )
            if type == 'insert_many':
                object = collection.data.insert_many(
                    **kwargs
                )
            if type == 'delete':
                object = collection.data.delete_by_id(
                    **kwargs
                )
            if type == 'delete_many':
                from weaviate.classes.query import Filter
                # TODO - this can be heavily expanded upon - leaving it bare bones as I cant foresee automated systems using this all!
                object = collection.data.delete_many(
                    where=Filter.by_property(name=delete_many_filter['name']).equal(delete_many_filter['lookup'])
                )
            if type == 'update':
                collection.data.update(
                    **kwargs
                )
            if type == 'reference':
                if update_options == 'add':
                    collection.data.reference_add(
                    **kwargs
                    )
                if update_options == 'many':
                    object = collection.data.reference_add_many(
                            **kwargs
                    )
                if update_options == 'delete':
                    collection.data.reference_delete(
                        **kwargs
                    )
                if update_options == 'update':
                    collection.data.reference_replace(
                        **kwargs
                    )
        except WeaviateQueryError:
            raise WeaviateQueryError
        return object if object else None

    def close_connection(
        self
    ) -> None:
        self.client.close()
        





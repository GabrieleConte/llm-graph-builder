from langchain_core.messages import AIMessage
from .src.main import *
from .src.shared.llm_graph_builder_exception import LLMGraphBuilderException
from .src.graphDB_dataAccess import graphDBdataAccess
from typing import Optional
from .src.response_format import create_response
from .src.QA_integration import *
from .src.shared.common_fn import *
import asyncio
from .src.logger import CustomLogger
from datetime import datetime, timezone
import gc
from langchain_neo4j import Neo4jGraph
from .src.post_processing import create_entity_embedding, create_vector_fulltext_indexes


logger = CustomLogger()
MERGED_DIR = os.path.join(os.path.dirname(__file__), "merged_files")


# ---------------------UTILITY FUNCTIONS---------------------
def sanitize_filename(filename):
    """
   Sanitize the user-provided filename to prevent directory traversal and remove unsafe characters.
   """
    # Remove path separators and collapse redundant separators
    filename = os.path.basename(filename)
    filename = os.path.normpath(filename)
    return filename


def validate_file_path(directory, filename):
    """
   Construct the full file path and ensure it is within the specified directory.
   """
    file_path = os.path.join(directory, filename)
    abs_directory = os.path.abspath(directory)
    abs_file_path = os.path.abspath(file_path)
    # Ensure the file path starts with the intended directory path
    if not abs_file_path.startswith(abs_directory):
        raise ValueError("Invalid file path")
    return abs_file_path


def compute_entity_embeddings(
        uri: str = "bolt://localhost:7687",
        userName: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
):
    """
    Computes embeddings for entities w/o one in the graph database.
    """
    graph = create_graph_database_connection(uri, userName, password, database)
    graphDBdataAccess(graph)

    create_entity_embedding(graph)


# ---------------------KG BUILDER FUNCTIONS---------------------
# Function to execute cypher query on KG
async def execute_cypher_query(
        uri: str = "bolt://localhost:7687",
        userName: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        query: str = None,
        params: Optional[dict] = None,
):
    try:
        graph = create_graph_database_connection(uri, userName, password, database)
        result = await asyncio.to_thread(execute_graph_query, graph, query, params, max_retries=3, delay=2)
        return create_response('Success', data=result, message='Query executed successfully')

    except Exception as e:
        job_status = "Failed"
        message = "Unable to execute query"
        error_message = str(e)
        logging.info(message)
        logging.exception(f'Exception:{error_message}')
        return create_response(job_status, message=message + error_message[:100], error=error_message)


# Function to upload files
async def upload_file(
        uri: str = "bolt://localhost:7687",
        userName: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        model_name: str = None,
        fileName: str = None,
):
    try:
        graph = create_graph_database_connection(uri, userName, password, database)
        result = await asyncio.to_thread(upload_file_on_db, graph, model_name, fileName)
        return create_response('Success', data=result, message='Source Node Created Successfully')

    except Exception as e:
        message = "Unable to upload file"
        error_message = str(e)
        graph = create_graph_database_connection(uri, userName, password, database)
        graphDb_data_Access = graphDBdataAccess(graph)
        graphDb_data_Access.update_exception_db(fileName, error_message)
        logging.info(message)
        logging.exception(f'Exception:{error_message}')
        return create_response('Failed', message=message + error_message[:100], error=error_message)
    finally:
        gc.collect()


#Function to delete document and its entities from graph database
async def delete_document_and_entities(
        uri: str = "bolt://localhost:7687",
        userName: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        fileName: str = None,
):
    fileName = f'["{fileName}"]'
    source_type = '["None"]'
    try:
        graph = create_graph_database_connection(uri, userName, password, database)
        graphDb_data_Access = graphDBdataAccess(graph)
        files_list_size = await asyncio.to_thread(graphDb_data_Access.delete_file_from_graph, fileName, source_type,
                                                  deleteEntities="true", merged_dir=MERGED_DIR)
        await asyncio.to_thread(graphDb_data_Access.delete_unconnected_nodes)
        message = f"Deleted {files_list_size} documents with entities from database"
        logging.info(message)
        return create_response('Success', message=message)

    except Exception as e:
        job_status = "Failed"
        message = f"Unable to delete document {fileName}"
        error_message = str(e)
        logging.exception(f'{message}:{error_message}')
        return create_response(job_status, message=message, error=error_message)
    finally:
        gc.collect()


# Function to extract KG from a local file
async def extract_knowledge_graph_from_file(
        uri: str = "bolt://localhost:7687",
        userName: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        model_env_value: str = None,
        model_name: str = None,
        file_name: str = None,
        content=None,
        allowedNodes: str = None,
        allowedRelationship: str = None,
        token_chunk_size: int = 300,
        chunk_overlap: int = 20,
        chunks_to_combine: int = 3,
        max_token_chunk_size: int = 500000,
        words_for_big_file: int = 2000,
        retry_condition: str = None,
        additional_instructions: Optional[str] = None,

):
    try:
        start_time = time.time()
        graph = create_graph_database_connection(uri, userName, password, database)
        graphDBdataAccess(graph)

        file_name = sanitize_filename(file_name)

        if not os.path.exists(MERGED_DIR):
            os.makedirs(MERGED_DIR)

        merged_file_path = validate_file_path(MERGED_DIR, file_name)

        with open(merged_file_path, "wb") as f:
            f.write(content)

        uri_latency, result = await extract_graph_from_file_local_file(uri, userName, password, database,
                                                                       model_env_value, model_name,
                                                                       merged_file_path, file_name, allowedNodes,
                                                                       allowedRelationship, token_chunk_size,
                                                                       chunk_overlap, chunks_to_combine,
                                                                       max_token_chunk_size, words_for_big_file,
                                                                       retry_condition, additional_instructions)

        extract_api_time = time.time() - start_time

        if result is not None:
            logging.info("Going for counting nodes and relationships in extract")
            count_node_time = time.time()
            graph = create_graph_database_connection(uri, userName, password, database)
            graphDb_data_Access = graphDBdataAccess(graph)
            count_response = graphDb_data_Access.update_node_relationship_count(file_name)
            logging.info("Nodes and Relationship Counts updated")
            if count_response:
                result['chunkNodeCount'] = count_response[file_name].get('chunkNodeCount', "0")
                result['chunkRelCount'] = count_response[file_name].get('chunkRelCount', "0")
                result['entityNodeCount'] = count_response[file_name].get('entityNodeCount', "0")
                result['entityEntityRelCount'] = count_response[file_name].get('entityEntityRelCount', "0")
                result['communityNodeCount'] = count_response[file_name].get('communityNodeCount', "0")
                result['communityRelCount'] = count_response[file_name].get('communityRelCount', "0")
                result['nodeCount'] = count_response[file_name].get('nodeCount', "0")
                result['relationshipCount'] = count_response[file_name].get('relationshipCount', "0")
                logging.info(f"counting completed in {(time.time() - count_node_time):.2f}")
            result['db_url'] = uri
            result['api_name'] = 'extract'
            result['logging_time'] = formatted_time(datetime.now(timezone.utc))
            result['elapsed_api_time'] = f'{extract_api_time:.2f}'
            result['userName'] = userName
            result['database'] = database
            result['retry_condition'] = retry_condition
        logging.info(f"extraction completed in {extract_api_time:.2f} seconds for file name {file_name}")

        return create_response('Success', data=result)

    except LLMGraphBuilderException as e:
        error_message = str(e)
        graph = create_graph_database_connection(uri, userName, password, database)
        graphDb_data_Access = graphDBdataAccess(graph)
        graphDb_data_Access.update_exception_db(file_name, error_message, retry_condition)

        failed_file_process(file_name, merged_file_path)
        graphDb_data_Access.get_current_status_document_node(file_name)
        logging.exception(f'File Failed in extraction: {e}')
        return create_response("Failed", message=error_message, error=error_message, file_name=file_name)

    except Exception as e:
        message = f"Failed To Process File:{file_name} or LLM Unable To Parse Content "
        error_message = str(e)
        graph = create_graph_database_connection(uri, userName, password, database)
        graphDb_data_Access = graphDBdataAccess(graph)
        graphDb_data_Access.update_exception_db(file_name, error_message, retry_condition)

        failed_file_process(file_name, merged_file_path)
        graphDb_data_Access.get_current_status_document_node(file_name)
        logging.exception(f'File Failed in extraction: {e}')
        return create_response('Failed', message=message + error_message[:100], error=error_message,
                               file_name=file_name)
    finally:
        gc.collect()


#Function to post-process the graph database after KG extraction
async def post_processing(
        uri: str = "bolt://localhost:7687",
        userName: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
):
    try:
        graph = create_graph_database_connection(uri, userName, password, database)

        # materialize_text_chunk_similarities: updates the graph node with SIMILAR relationship where embedding score match
        await asyncio.to_thread(update_graph, graph)
        logging.info(f'Updated KNN Graph')

        # enable_hybrid_search_and_fulltext_search_in_bloom: creates full text index on the graph for bloom search
        await asyncio.to_thread(create_vector_fulltext_indexes, uri=uri, username=userName, password=password,
                                database=database)
        logging.info(f'Full Text index created')

        # materialize_entity_similarities: creates entity embeddings
        await asyncio.to_thread(create_entity_embedding, graph)
        logging.info(f'Entity Embeddings created')

        graph = create_graph_database_connection(uri, userName, password, database)
        graphDb_data_Access = graphDBdataAccess(graph)
        document_name = ""
        count_response = graphDb_data_Access.update_node_relationship_count(document_name)
        if count_response:
            count_response = [{"filename": filename, **counts} for filename, counts in count_response.items()]
            logging.info(f'Updated source node with community related counts')

        return create_response('Success', data=count_response, message='All tasks completed successfully')

    except Exception as e:
        job_status = "Failed"
        error_message = str(e)
        message = f"Unable to complete tasks"
        logging.exception(f'Exception in post_processing tasks: {error_message}')
        return create_response(job_status, message=message, error=error_message)

    finally:
        gc.collect()


async def chat_bot(
        uri: str = "bolt://localhost:7687",
        userName: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        model_env_value: str = None,
        model_name: str = None,
        history=None,
        question: str = None,
        mode: str = "graph_vector_fulltext",
        embedding_model: str = None,
):
    logging.info(f"QA_RAG called at {datetime.now()}")
    qa_rag_start_time = time.time()
    document_names = '[]'

    try:
        if mode == "graph":
            graph = Neo4jGraph(url=uri, username=userName, password=password, database=database, sanitize=True,
                               refresh_schema=True)
        else:
            graph = create_graph_database_connection(uri, userName, password, database)

        graphDBdataAccess(graph)
        result = await asyncio.to_thread(QA_RAG, graph=graph, model_env_value=model_env_value, model_name=model_name,
                                         history=history, question=question, document_names=document_names, mode=mode, embedding_model=embedding_model)

        total_call_time = time.time() - qa_rag_start_time
        logging.info(f"Success: total Response time is  {total_call_time:.2f} seconds")

        return result

    except Exception as e:
        logging.exception(f'Exception in chat bot:{str(e)}')
        return "Failed: unable to get chat response"
    finally:
        gc.collect()


async def main():
    # ENVIRONMENT VARIABLES
    uri = "bolt://localhost:7687"
    userName = "neo4j"
    password = "password"
    database = "neo4j"
    model_env_value = "groq,openai/gpt-oss-120b,(...)"
    model_name = "openai/gpt-oss-120b"

    # ---------------------FARE DOMANDE AL CHATBOT---------------------
    messages = [
        AIMessage(content="Hello! How can I assist you today?"),
        HumanMessage(content="Tell me info about stingray"),
        AIMessage(
            content="The 1963 Chevrolet Corvette Sting Ray is a two‑door coupe produced by the American automaker Chevrolet. It features a fastback rear design, hidden headlamps, and distinctive “humps” over the fenders, along with a split‑back rear window (a design later changed for safety). In 1963 Chevrolet built about 21,000 Sting Rays. Performance specs listed in the source are a top speed of 118 mph, 0‑60 mph in 6.1 seconds, and fuel consumption of 18 mpg."),
    ]

    result = await chat_bot(
        question="is it a car from 60s?",
        history=messages,
        uri=uri,
        userName=userName,
        password=password,
        database=database,
        model_env_value=model_env_value,
        model_name=model_name,
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())

from src.shared.constants import (QUERY_TO_GET_CHUNKS,
                                  QUERY_TO_DELETE_EXISTING_ENTITIES,
                                  QUERY_TO_GET_LAST_PROCESSED_CHUNK_POSITION,
                                  QUERY_TO_GET_LAST_PROCESSED_CHUNK_WITHOUT_ENTITY,
                                  START_FROM_BEGINNING,
                                  START_FROM_LAST_PROCESSED_POSITION,
                                  DELETE_ENTITIES_AND_START_FROM_BEGINNING,
                                  QUERY_TO_GET_NODES_AND_RELATIONS_OF_A_DOCUMENT)
from datetime import datetime
from src.create_chunks import CreateChunksofDocument
from src.graphDB_dataAccess import graphDBdataAccess
from src.local_file import get_documents_from_file_by_path
from src.source_node import sourceNode
from src.llm import get_graph_from_llm
from src.shared.common_fn import *
from src.make_relationships import *
import warnings
from src.shared.llm_graph_builder_exception import LLMGraphBuilderException
from langchain.docstore.document import Document

warnings.filterwarnings("ignore")
logging.basicConfig(format='%(asctime)s - %(message)s', level='INFO')


async def extract_graph_from_file_local_file(uri, userName, password, database, model_env_value, model_name,
                                             merged_file_path, fileName,
                                             allowedNodes, allowedRelationship, token_chunk_size, chunk_overlap,
                                             chunks_to_combine, max_token_chunk_size, words_for_big_file,
                                             retry_condition,
                                             additional_instructions):
    logging.info(f'Process file name :{fileName}')

    if not retry_condition:
        file_name, pages, file_extension = get_documents_from_file_by_path(merged_file_path, fileName)

        total_file_words = sum(len(p.page_content.split()) for p in pages)
        big_file = (
                    total_file_words > words_for_big_file)  # True if the file is considered big (do not perform entity extraction)

        if pages == None or len(pages) == 0:
            raise LLMGraphBuilderException(f'File content is not available for file : {file_name}')
        return await processing_source(uri, userName, password, database, model_env_value, model_name, file_name, pages,
                                       allowedNodes,
                                       allowedRelationship, token_chunk_size, chunk_overlap, chunks_to_combine,
                                       max_token_chunk_size,
                                       True, merged_file_path=merged_file_path, retry_condition=retry_condition,
                                       additional_instructions=additional_instructions,
                                       big_file=big_file)
    else:
        return await processing_source(uri, userName, password, database, model_env_value, model_name, fileName, [],
                                       allowedNodes,
                                       allowedRelationship, token_chunk_size, chunk_overlap, chunks_to_combine,
                                       max_token_chunk_size,
                                       True, merged_file_path=merged_file_path, retry_condition=retry_condition,
                                       additional_instructions=additional_instructions, big_file=False)


async def processing_source(uri, userName, password, database, model_env_value, model_name, file_name, pages,
                            allowedNodes,
                            allowedRelationship, token_chunk_size, chunk_overlap, chunks_to_combine,
                            max_token_chunk_size,
                            is_uploaded_from_local=None, merged_file_path=None, retry_condition=None,
                            additional_instructions=None, big_file=False):
    """
   Extracts a Neo4jGraph from a PDF file based on the model.
   
   Args:
   	 uri: URI of the graph to extract
     db_name : db_name is database name to connect graph db
   	 userName: Username to use for graph creation ( if None will use username from config file )
   	 password: Password to use for graph creation ( if None will use password from config file )
   	 file: File object containing the PDF file to be used
   	 model: Type of model to use ('Diffbot'or'OpenAI GPT')
   
   Returns: 
   	 Json response to API with fileName, nodeCount, relationshipCount, processingTime, 
     status and model as attributes.
  """
    uri_latency = {}
    response = {}
    start_time = datetime.now()
    processing_source_start_time = time.time()

    start_create_connection = time.time()
    graph = create_graph_database_connection(uri, userName, password, database)
    end_create_connection = time.time()

    elapsed_create_connection = end_create_connection - start_create_connection
    logging.info(f'Time taken database connection: {elapsed_create_connection:.2f} seconds')
    uri_latency["create_connection"] = f'{elapsed_create_connection:.2f}'
    graphDb_data_Access = graphDBdataAccess(graph)

    create_chunk_vector_index(graph)

    start_get_chunkId_chunkDoc_list = time.time()
    total_chunks, chunkId_chunkDoc_list = get_chunkId_chunkDoc_list(graph, file_name, pages, token_chunk_size,
                                                                    chunk_overlap, max_token_chunk_size,
                                                                    retry_condition)
    end_get_chunkId_chunkDoc_list = time.time()

    elapsed_get_chunkId_chunkDoc_list = end_get_chunkId_chunkDoc_list - start_get_chunkId_chunkDoc_list
    logging.info(
        f'Time taken to create list chunkids with chunk document: {elapsed_get_chunkId_chunkDoc_list:.2f} seconds')
    uri_latency["create_list_chunk_and_document"] = f'{elapsed_get_chunkId_chunkDoc_list:.2f}'
    uri_latency["total_chunks"] = total_chunks

    start_status_document_node = time.time()
    result = graphDb_data_Access.get_current_status_document_node(file_name)
    end_status_document_node = time.time()

    elapsed_status_document_node = end_status_document_node - start_status_document_node
    logging.info(f'Time taken to get the current status of document node: {elapsed_status_document_node:.2f} seconds')
    uri_latency["get_status_document_node"] = f'{elapsed_status_document_node:.2f}'

    select_chunks_with_retry = 0
    node_count = 0
    rel_count = 0

    if len(result) > 0:
        if result[0]['Status'] != 'Processing':
            obj_source_node = sourceNode()
            status = "Processing"
            obj_source_node.file_name = file_name.strip() if isinstance(file_name, str) else file_name
            obj_source_node.status = status
            obj_source_node.total_chunks = total_chunks
            obj_source_node.model = model_name
            if retry_condition == START_FROM_LAST_PROCESSED_POSITION:
                node_count = result[0]['nodeCount']
                rel_count = result[0]['relationshipCount']
                select_chunks_with_retry = result[0]['processed_chunk']
            obj_source_node.processed_chunk = 0 + select_chunks_with_retry
            logging.info(file_name)
            logging.info(obj_source_node)

            start_update_source_node = time.time()
            graphDb_data_Access.update_source_node(obj_source_node)
            graphDb_data_Access.update_node_relationship_count(file_name)
            end_update_source_node = time.time()

            elapsed_update_source_node = end_update_source_node - start_update_source_node
            logging.info(f'Time taken to update the document source node: {elapsed_update_source_node:.2f} seconds')
            uri_latency["update_source_node"] = f'{elapsed_update_source_node:.2f}'

            logging.info('Update the status as Processing')
            update_graph_chunk_processed = 20  # BATCH OF CHUNK IN THE PROCESSING LOOP
            job_status = "Completed"

            if big_file:
                logging.info(
                    f"Big file with more than 2000 words, hence extracting only chunk nodes - skipping graph documents and entity extraction for file {file_name}")

            for i in range(0, len(chunkId_chunkDoc_list), update_graph_chunk_processed):
                select_chunks_upto = i + update_graph_chunk_processed
                logging.info(f'Selected Chunks upto: {select_chunks_upto}')
                if len(chunkId_chunkDoc_list) <= select_chunks_upto:
                    select_chunks_upto = len(chunkId_chunkDoc_list)
                selected_chunks = chunkId_chunkDoc_list[i:select_chunks_upto]

                result = graphDb_data_Access.get_current_status_document_node(file_name)
                is_cancelled_status = result[0]['is_cancelled']
                logging.info(f"Value of is_cancelled : {result[0]['is_cancelled']}")
                if bool(is_cancelled_status) == True:
                    job_status = "Cancelled"
                    logging.info('Exit from running loop of processing file')
                    break
                else:
                    processing_chunks_start_time = time.time()

                    node_count, rel_count, latency_processed_chunk = await processing_chunks(selected_chunks, graph,
                                                                                             uri, userName, password,
                                                                                             database, file_name,
                                                                                             model_env_value,
                                                                                             allowedNodes,
                                                                                             allowedRelationship,
                                                                                             chunks_to_combine,
                                                                                             additional_instructions,
                                                                                             big_file=big_file)

                    processing_chunks_end_time = time.time()
                    processing_chunks_elapsed_end_time = processing_chunks_end_time - processing_chunks_start_time
                    logging.info(
                        f"Time taken {update_graph_chunk_processed} chunks processed upto {select_chunks_upto} completed in {processing_chunks_elapsed_end_time:.2f} seconds for file name {file_name}")
                    uri_latency[
                        f'processed_combine_chunk_{i}-{select_chunks_upto}'] = f'{processing_chunks_elapsed_end_time:.2f}'
                    uri_latency[f'processed_chunk_detail_{i}-{select_chunks_upto}'] = latency_processed_chunk
                    end_time = datetime.now()
                    processed_time = end_time - start_time

                    obj_source_node = sourceNode()
                    obj_source_node.file_name = file_name
                    obj_source_node.updated_at = end_time
                    obj_source_node.processing_time = processed_time
                    obj_source_node.processed_chunk = select_chunks_upto + select_chunks_with_retry
                    if retry_condition == START_FROM_BEGINNING:
                        result = execute_graph_query(graph, QUERY_TO_GET_NODES_AND_RELATIONS_OF_A_DOCUMENT,
                                                     params={"filename": file_name})
                        obj_source_node.node_count = result[0]['nodes']
                        obj_source_node.relationship_count = result[0]['rels']
                    else:
                        obj_source_node.node_count = node_count
                        obj_source_node.relationship_count = rel_count
                    graphDb_data_Access.update_source_node(obj_source_node)
                    graphDb_data_Access.update_node_relationship_count(file_name)

            result = graphDb_data_Access.get_current_status_document_node(file_name)
            is_cancelled_status = result[0]['is_cancelled']
            if bool(is_cancelled_status) == True:
                logging.info(f'Is_cancelled True at the end extraction')
                job_status = 'Cancelled'
            logging.info(f'Job Status at the end : {job_status}')
            end_time = datetime.now()
            processed_time = end_time - start_time
            obj_source_node = sourceNode()
            obj_source_node.file_name = file_name.strip() if isinstance(file_name, str) else file_name
            obj_source_node.status = job_status
            obj_source_node.processing_time = processed_time

            graphDb_data_Access.update_source_node(obj_source_node)
            graphDb_data_Access.update_node_relationship_count(file_name)
            logging.info('Updated the nodeCount and relCount properties in Document node')
            logging.info(f'file:{file_name} extraction has been completed')

            if is_uploaded_from_local:
                delete_uploaded_local_file(merged_file_path, file_name)

            processing_source_func = time.time() - processing_source_start_time
            logging.info(
                f"Time taken to processing source function completed in {processing_source_func:.2f} seconds for file name {file_name}")
            uri_latency["Processed_source"] = f'{processing_source_func:.2f}'
            if node_count == 0:
                uri_latency["Per_entity_latency"] = 'N/A'
            else:
                uri_latency["Per_entity_latency"] = f'{int(processing_source_func) / node_count}/s'

            response["fileName"] = file_name
            response["nodeCount"] = node_count
            response["relationshipCount"] = rel_count
            response["total_processing_time"] = round(processed_time.total_seconds(), 2)
            response["status"] = job_status
            response["model"] = model_name
            response["success_count"] = 1

            return uri_latency, response
        else:
            logging.info("File does not process because its already in Processing status")
            return uri_latency, response
    else:
        error_message = "Unable to get the status of document node."
        logging.error(error_message)
        raise LLMGraphBuilderException(error_message)


async def processing_chunks(chunkId_chunkDoc_list, graph, uri, userName, password, database, file_name, model_env_value,
                            allowedNodes, allowedRelationship, chunks_to_combine, additional_instructions=None,
                            big_file=False):
    #create vector index and update chunk node with embedding
    latency_processing_chunk = {}
    if graph is not None:
        if graph._driver._closed:
            graph = create_graph_database_connection(uri, userName, password, database)
    else:
        graph = create_graph_database_connection(uri, userName, password, database)

    start_update_embedding = time.time()
    create_chunk_embeddings(graph, chunkId_chunkDoc_list, file_name)

    end_update_embedding = time.time()
    elapsed_update_embedding = end_update_embedding - start_update_embedding
    logging.info(f'Time taken to update embedding in chunk node: {elapsed_update_embedding:.2f} seconds')
    latency_processing_chunk["update_embedding"] = f'{elapsed_update_embedding:.2f}'

    if big_file:
        graphDb_data_Access = graphDBdataAccess(graph)
        count_response = graphDb_data_Access.update_node_relationship_count(file_name)
        node_count = count_response[file_name].get('nodeCount', "0")
        rel_count = count_response[file_name].get('relationshipCount', "0")
        return node_count, rel_count, latency_processing_chunk

    logging.info("Get graph document list from models")

    if "phoneCall_" in file_name:
        allowedNodes = "Contact,Date,Duration,Time,App,Call"
        allowedRelationship = "Call,WITH_CONTACT,Contact,Call,HAS_DATE,Date,Call,SOURCE_APP,App,Call,HAS_DURATION,Duration,Call,HAS_START_TIME,Time,Call,HAS_END_TIME,Time"

    elif "contact_" in file_name:
        allowedNodes = "Contact,Phone_Number,App,Name"
        allowedRelationship = "Contact,SOURCE_APP,App,Contact,HAS_NAME,Name,Contact,HAS_PHONE_NUMBER,Phone_Number"

    elif "alarm_" in file_name:
        allowedNodes = "App,Alarm,Recurrence_Type,Label,Time,Date"
        allowedRelationship = "Alarm,SOURCE_APP,App,Alarm,HAS_TYPE,Recurrence_Type,Alarm,HAS_LABEL,Label,Alarm,HAS_TIME,Time,Alarm,HAS_DATE,Date"

    elif "recurrentAlarm_" in file_name:
        allowedNodes = "App,Alarm,Alarm_Type,Label,Repeat_Frequency,Time,Day_Of_Week,Day_Of_Month,Date"
        allowedRelationship = "Alarm,SOURCE_APP,App,Alarm,HAS_TYPE,Alarm_Type,Alarm,HAS_LABEL,Label,Alarm,HAS_TIME,Time,Alarm,HAS_FREQUENCY,Repeat_Frequency,Alarm,SET_ON,Date,Alarm,SET_ON,Day_Of_Week,Alarm,SET_ON,Day_Of_Month"

    elif "event_" in file_name:
        allowedNodes = "App,Event,Recurrence_Type,Label,Time,Date"
        allowedRelationship = "Event,SOURCE_APP,App,Event,HAS_TYPE,Recurrence_Type,Event,HAS_LABEL,Label,Event,HAS_START_TIME,Time,Event,HAS_END_TIME,Time,Event,HAS_DATE,Date"

    elif "recurrentEvent_" in file_name:
        allowedNodes = "App,Event,Recurrence_Type,Label,Repeat_Frequency,Time,Day_Of_Week,Day_Of_Month,Date"
        allowedRelationship = "Event,SOURCE_APP,App,Event,HAS_TYPE,Recurrence_Type,Event,HAS_LABEL,Label,Event,HAS_START_TIME,Time,Event,HAS_END_TIME,Time,Event,HAS_FREQUENCY,Repeat_Frequency,Event,SET_ON,Date,Event,SET_ON,Day_Of_Week,Event,SET_ON,Day_Of_Month"

    elif "note_" in file_name:
        allowedNodes = "App,Note,Label,Date,Time"
        allowedRelationship = "Note,SOURCE_APP,App,Note,HAS_LABEL,Label,Note,CREATION_DATE,Date,Note,CREATION_TIME,Time,Note,TIME_MODIFIED,Time,Note,DATE_MODIFIED,Date"

    elif "photo_" in file_name:
        allowedNodes = "App,Photo,Location,Date,Time,Path"
        allowedRelationship = "Photo,SOURCE_APP,App,Photo,CREATION_DATE,Date,Photo,CREATION_TIME,Time,Photo,TAKEN_IN,Location,Photo,HAS_PATH,Path"

    elif "doc_" in file_name:
        allowedNodes = "App,Doc,Date,Time,Path"
        allowedRelationship = "Doc,SOURCE_APP,App,Doc,CREATION_DATE,Date,Doc,CREATION_TIME,Time,Doc,DATE_MODIFIED,Date,Doc,TIME_MODIFIED,Time,Doc,HAS_PATH,Path"
    else:
        allowedNodes = ""
        allowedRelationship = ""

    start_entity_extraction = time.time()
    graph_documents = await get_graph_from_llm(model_env_value, chunkId_chunkDoc_list, allowedNodes,
                                               allowedRelationship,
                                               chunks_to_combine, additional_instructions)
    end_entity_extraction = time.time()

    elapsed_entity_extraction = end_entity_extraction - start_entity_extraction
    logging.info(f'Time taken to extract enitities from LLM Graph Builder: {elapsed_entity_extraction:.2f} seconds')
    latency_processing_chunk["entity_extraction"] = f'{elapsed_entity_extraction:.2f}'
    cleaned_graph_documents = handle_backticks_nodes_relationship_id_type(graph_documents)

    start_save_graphDocuments = time.time()
    save_graphDocuments_in_neo4j(graph, cleaned_graph_documents)
    end_save_graphDocuments = time.time()
    elapsed_save_graphDocuments = end_save_graphDocuments - start_save_graphDocuments
    logging.info(f'Time taken to save graph document in neo4j: {elapsed_save_graphDocuments:.2f} seconds')
    latency_processing_chunk["save_graphDocuments"] = f'{elapsed_save_graphDocuments:.2f}'

    chunks_and_graphDocuments_list = get_chunk_and_graphDocument(cleaned_graph_documents)

    start_relationship = time.time()
    merge_relationship_between_chunk_and_entites(graph, chunks_and_graphDocuments_list)
    end_relationship = time.time()
    elapsed_relationship = end_relationship - start_relationship
    logging.info(f'Time taken to create relationship between chunk and entities: {elapsed_relationship:.2f} seconds')
    latency_processing_chunk["relationship_between_chunk_entity"] = f'{elapsed_relationship:.2f}'

    graphDb_data_Access = graphDBdataAccess(graph)
    count_response = graphDb_data_Access.update_node_relationship_count(file_name)
    node_count = count_response[file_name].get('nodeCount', "0")
    rel_count = count_response[file_name].get('relationshipCount', "0")
    return node_count, rel_count, latency_processing_chunk


def get_chunkId_chunkDoc_list(graph, file_name, pages, token_chunk_size, chunk_overlap, max_token_chunk_size,
                              retry_condition):
    if not retry_condition:
        logging.info("Break down file into chunks")
        bad_chars = ['"', "\n", "'"]
        for i in range(0, len(pages)):
            text = pages[i].page_content
            for j in bad_chars:
                if j == '\n':
                    text = text.replace(j, ' ')
                else:
                    text = text.replace(j, '')
            pages[i] = Document(page_content=str(text), metadata=pages[i].metadata)
        create_chunks_obj = CreateChunksofDocument(pages, graph)
        chunks = create_chunks_obj.split_file_into_chunks(token_chunk_size, chunk_overlap, max_token_chunk_size)
        chunkId_chunkDoc_list = create_relation_between_chunks(graph, file_name, chunks)
        return len(chunks), chunkId_chunkDoc_list

    else:
        chunkId_chunkDoc_list = []
        chunks = execute_graph_query(graph, QUERY_TO_GET_CHUNKS, params={"filename": file_name})

        if chunks[0]['text'] is None or chunks[0]['text'] == "" or not chunks:
            raise LLMGraphBuilderException(
                f"Chunks are not created for {file_name}. Please re-upload file and try again.")
        else:
            for chunk in chunks:
                chunk_doc = Document(page_content=chunk['text'],
                                     metadata={'id': chunk['id'], 'position': chunk['position']})
                chunkId_chunkDoc_list.append({'chunk_id': chunk['id'], 'chunk_doc': chunk_doc})

            if retry_condition == START_FROM_LAST_PROCESSED_POSITION:
                logging.info(f"Retry : start_from_last_processed_position")
                starting_chunk = execute_graph_query(graph, QUERY_TO_GET_LAST_PROCESSED_CHUNK_POSITION,
                                                     params={"filename": file_name})

                if starting_chunk and starting_chunk[0]["position"] < len(chunkId_chunkDoc_list):
                    return len(chunks), chunkId_chunkDoc_list[starting_chunk[0]["position"] - 1:]

                elif starting_chunk and starting_chunk[0]["position"] == len(chunkId_chunkDoc_list):
                    starting_chunk = execute_graph_query(graph, QUERY_TO_GET_LAST_PROCESSED_CHUNK_WITHOUT_ENTITY,
                                                         params={"filename": file_name})
                    return len(chunks), chunkId_chunkDoc_list[starting_chunk[0]["position"] - 1:]

                else:
                    raise LLMGraphBuilderException(
                        f"All chunks of file {file_name} are already processed. If you want to re-process, Please start from begnning")

            else:
                logging.info(f"Retry : start_from_beginning with chunks {len(chunkId_chunkDoc_list)}")
                return len(chunks), chunkId_chunkDoc_list


def update_graph(graph):
    """
  Update the graph node with SIMILAR relationship where embedding score match
  """
    graph_DB_dataAccess = graphDBdataAccess(graph)
    graph_DB_dataAccess.update_KNN_graph()


def connection_check_and_get_vector_dimensions(graph, database):
    """
  Args:
    uri: URI of the graph to extract
    userName: Username to use for graph creation ( if None will use username from config file )
    password: Password to use for graph creation ( if None will use password from config file )
    db_name: db_name is database name to connect to graph db
  Returns:
   Returns a status of connection from NEO4j is success or failure
 """
    graph_DB_dataAccess = graphDBdataAccess(graph)
    return graph_DB_dataAccess.connection_check_and_get_vector_dimensions(database)


def upload_file_on_db(graph, model_name, fileName):
    obj_source_node = sourceNode()
    obj_source_node.file_name = fileName
    obj_source_node.file_type = fileName.split('.')[-1]
    obj_source_node.file_size = 0
    obj_source_node.file_source = 'None'
    obj_source_node.model = model_name
    obj_source_node.created_at = datetime.now()
    obj_source_node.chunkNodeCount = 0
    obj_source_node.chunkRelCount = 0
    obj_source_node.entityNodeCount = 0
    obj_source_node.entityEntityRelCount = 0
    obj_source_node.communityNodeCount = 0
    obj_source_node.communityRelCount = 0
    graphDb_data_Access = graphDBdataAccess(graph)

    graphDb_data_Access.create_source_node(obj_source_node)


def failed_file_process(file_name, merged_file_path):
    logging.info(f'Deleted File Path: {merged_file_path} and Deleted File Name : {file_name}')
    delete_uploaded_local_file(merged_file_path, file_name)

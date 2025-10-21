import logging
import os
import time
from neo4j.exceptions import TransientError
from langchain_neo4j import Neo4jGraph
from .shared.common_fn import delete_uploaded_local_file
from .shared.constants import NODEREL_COUNT_QUERY_WITH_COMMUNITY, NODEREL_COUNT_QUERY_WITHOUT_COMMUNITY
from .source_node import sourceNode
from .communities import MAX_COMMUNITY_LEVELS
import json


class graphDBdataAccess:

    def __init__(self, graph: Neo4jGraph):
        self.graph = graph

    def update_exception_db(self, file_name, exp_msg, retry_condition=None):
        try:
            job_status = "Failed"
            result = self.get_current_status_document_node(file_name)
            if len(result) > 0:
                is_cancelled_status = result[0]['is_cancelled']
                if bool(is_cancelled_status) == True:
                    job_status = 'Cancelled'
            if retry_condition is not None:
                retry_condition = None
                self.graph.query(
                    """MERGE(d:Document {fileName :$fName}) SET d.status = $status, d.errorMessage = $error_msg, d.retry_condition = $retry_condition""",
                    {"fName": file_name, "status": job_status, "error_msg": exp_msg,
                     "retry_condition": retry_condition}, session_params={"database": self.graph._database})
            else:
                self.graph.query(
                    """MERGE(d:Document {fileName :$fName}) SET d.status = $status, d.errorMessage = $error_msg""",
                    {"fName": file_name, "status": job_status, "error_msg": exp_msg},
                    session_params={"database": self.graph._database})
        except Exception as e:
            error_message = str(e)
            logging.error(f"Error in updating document node status as failed: {error_message}")
            raise Exception(error_message)

    def create_source_node(self, obj_source_node: sourceNode):
        try:
            job_status = "New"
            logging.info(f"creating source node if does not exist in database {self.graph._database}")
            self.graph.query("""MERGE(d:Document {fileName :$fn}) SET d.fileSize = $fs, d.fileType = $ft ,
                            d.status = $st, d.url = $url, d.awsAccessKeyId = $awsacc_key_id, 
                            d.fileSource = $f_source, d.createdAt = $c_at, d.updatedAt = $u_at, 
                            d.processingTime = $pt, d.errorMessage = $e_message, d.nodeCount= $n_count, 
                            d.relationshipCount = $r_count, d.model= $model, d.gcsBucket=$gcs_bucket, 
                            d.gcsBucketFolder= $gcs_bucket_folder, d.language= $language,d.gcsProjectId= $gcs_project_id,
                            d.is_cancelled=False, d.total_chunks=0, d.processed_chunk=0,
                            d.access_token=$access_token,
                            d.chunkNodeCount=$chunkNodeCount,d.chunkRelCount=$chunkRelCount,
                            d.entityNodeCount=$entityNodeCount,d.entityEntityRelCount=$entityEntityRelCount,
                            d.communityNodeCount=$communityNodeCount,d.communityRelCount=$communityRelCount""",
                             {"fn": obj_source_node.file_name, "fs": obj_source_node.file_size,
                              "ft": obj_source_node.file_type, "st": job_status,
                              "url": obj_source_node.url,
                              "awsacc_key_id": obj_source_node.awsAccessKeyId, "f_source": obj_source_node.file_source,
                              "c_at": obj_source_node.created_at,
                              "u_at": obj_source_node.created_at, "pt": 0, "e_message": '', "n_count": 0, "r_count": 0,
                              "model": obj_source_node.model,
                              "gcs_bucket": obj_source_node.gcsBucket,
                              "gcs_bucket_folder": obj_source_node.gcsBucketFolder,
                              "language": obj_source_node.language, "gcs_project_id": obj_source_node.gcsProjectId,
                              "access_token": obj_source_node.access_token,
                              "chunkNodeCount": obj_source_node.chunkNodeCount,
                              "chunkRelCount": obj_source_node.chunkRelCount,
                              "entityNodeCount": obj_source_node.entityNodeCount,
                              "entityEntityRelCount": obj_source_node.entityEntityRelCount,
                              "communityNodeCount": obj_source_node.communityNodeCount,
                              "communityRelCount": obj_source_node.communityRelCount
                              }, session_params={"database": self.graph._database})
        except Exception as e:
            error_message = str(e)
            logging.info(f"error_message = {error_message}")
            self.update_exception_db(self, obj_source_node.file_name, error_message)
            raise Exception(error_message)

    def update_source_node(self, obj_source_node: sourceNode):
        try:

            params = {}
            if obj_source_node.file_name is not None and obj_source_node.file_name != '':
                params['fileName'] = obj_source_node.file_name

            if obj_source_node.status is not None and obj_source_node.status != '':
                params['status'] = obj_source_node.status

            if obj_source_node.created_at is not None:
                params['createdAt'] = obj_source_node.created_at

            if obj_source_node.updated_at is not None:
                params['updatedAt'] = obj_source_node.updated_at

            if obj_source_node.processing_time is not None and obj_source_node.processing_time != 0:
                params['processingTime'] = round(obj_source_node.processing_time.total_seconds(), 2)

            if obj_source_node.node_count is not None:
                params['nodeCount'] = obj_source_node.node_count

            if obj_source_node.relationship_count is not None:
                params['relationshipCount'] = obj_source_node.relationship_count

            if obj_source_node.model is not None and obj_source_node.model != '':
                params['model'] = obj_source_node.model

            if obj_source_node.total_chunks is not None and obj_source_node.total_chunks != 0:
                params['total_chunks'] = obj_source_node.total_chunks

            if obj_source_node.is_cancelled is not None:
                params['is_cancelled'] = obj_source_node.is_cancelled

            if obj_source_node.processed_chunk is not None:
                params['processed_chunk'] = obj_source_node.processed_chunk

            if obj_source_node.retry_condition is not None:
                params['retry_condition'] = obj_source_node.retry_condition

            param = {"props": params}

            logging.info(f'Base Param value 1 : {param}')
            query = "MERGE(d:Document {fileName :$props.fileName}) SET d += $props"
            logging.info("Update source node properties")
            self.graph.query(query, param, session_params={"database": self.graph._database})
        except Exception as e:
            error_message = str(e)
            self.update_exception_db(self, self.file_name, error_message)
            raise Exception(error_message)

    def update_KNN_graph(self):
        """
        Update the graph node with SIMILAR relationship where embedding score match
        """
        index = self.graph.query("""show indexes yield * where type = 'VECTOR' and name = 'vector'""",
                                 session_params={"database": self.graph._database})
        knn_min_score = 0.94
        if len(index) > 0:
            logging.info('update KNN graph')
            self.graph.query("""MATCH (c:Chunk)
                                    WHERE c.embedding IS NOT NULL AND count { (c)-[:SIMILAR]-() } < 5
                                    CALL db.index.vector.queryNodes('vector', 6, c.embedding) yield node, score
                                    WHERE node <> c and score >= $score MERGE (c)-[rel:SIMILAR]-(node) SET rel.score = score
                                """,
                             {"score": float(knn_min_score)}
                             , session_params={"database": self.graph._database})
        else:
            logging.info("Vector index does not exist, So KNN graph not update")

    def check_account_access(self, database):
        try:
            query_dbms_componenet = "call dbms.components() yield edition"
            result_dbms_componenet = self.graph.query(query_dbms_componenet,
                                                      session_params={"database": self.graph._database})

            if result_dbms_componenet[0]["edition"] == "enterprise":
                query = """
                SHOW USER PRIVILEGES 
                YIELD * 
                WHERE graph = $database AND action IN ['read'] 
                RETURN COUNT(*) AS readAccessCount
                """

                logging.info(f"Checking access for database: {database}")

                result = self.graph.query(query, params={"database": database},
                                          session_params={"database": self.graph._database})
                read_access_count = result[0]["readAccessCount"] if result else 0

                logging.info(f"Read access count: {read_access_count}")

                if read_access_count > 0:
                    logging.info("The account has read access.")
                    return False
                else:
                    logging.info("The account has write access.")
                    return True
            else:
                #Community version have no roles to execute admin command, so assuming write access as TRUE
                logging.info("The account has write access.")
                return True

        except Exception as e:
            logging.error(f"Error checking account access: {e}")
            return False

    def check_gds_version(self):
        try:
            gds_procedure_count = """
            SHOW FUNCTIONS YIELD name WHERE name STARTS WITH 'gds.version' RETURN COUNT(*) AS totalGdsProcedures
            """
            result = self.graph.query(gds_procedure_count, session_params={"database": self.graph._database})
            total_gds_procedures = result[0]['totalGdsProcedures'] if result else 0

            if total_gds_procedures > 0:
                logging.info("GDS is available in the database.")
                return True
            else:
                logging.info("GDS is not available in the database.")
                return False
        except Exception as e:
            logging.error(f"An error occurred while checking GDS version: {e}")
            return False

    def connection_check_and_get_vector_dimensions(self, database, embedding_model):
        """
        Get the vector index dimension from database and application configuration and DB connection status
        
        Args:
            uri: URI of the graph to extract
            userName: Username to use for graph creation ( if None will use username from config file )
            password: Password to use for graph creation ( if None will use password from config file )
            db_name: db_name is database name to connect to graph db
        Returns:
        Returns a status of connection from NEO4j is success or failure
        """

        db_vector_dimension = self.graph.query("""SHOW INDEXES YIELD *
                                    WHERE type = 'VECTOR' AND name = 'vector'
                                    RETURN options.indexConfig['vector.dimensions'] AS vector_dimensions
                                """, session_params={"database": self.graph._database})

        result_chunks = self.graph.query("""match (c:Chunk) return size(c.embedding) as embeddingSize, count(*) as chunks, 
                                                    count(c.embedding) as hasEmbedding
                                """, session_params={"database": self.graph._database})


        logging.info(f'embedding model:{embedding_model}')

        gds_status = self.check_gds_version()
        write_access = self.check_account_access(database=database)

        if self.graph:
            if len(db_vector_dimension) > 0:
                return {'db_vector_dimension': db_vector_dimension[0]['vector_dimensions'],
                        'message': "Connection Successful",
                        "gds_status": gds_status, "write_access": write_access}
            else:
                if len(db_vector_dimension) == 0 and len(result_chunks) == 0:
                    logging.info("Chunks and vector index does not exists in database")
                    return {'db_vector_dimension': 0, 'message': "Connection Successful", "chunks_exists": False,
                            "gds_status": gds_status,
                            "write_access": write_access}
                elif len(db_vector_dimension) == 0 and result_chunks[0]['hasEmbedding'] == 0 and result_chunks[0]['chunks'] > 0:
                    return {'db_vector_dimension': 0, 'message': "Connection Successful", "chunks_exists": True,
                            "gds_status": gds_status,
                            "write_access": write_access}
                else:
                    return {'message': "Connection Successful", "gds_status": gds_status, "write_access": write_access}

    def execute_query(self, query, param=None, max_retries=3, delay=2):
        retries = 0
        while retries < max_retries:
            try:
                return self.graph.query(query, param, session_params={"database": self.graph._database})
            except TransientError as e:
                if "DeadlockDetected" in str(e):
                    retries += 1
                    logging.info(f"Deadlock detected. Retrying {retries}/{max_retries} in {delay} seconds...")
                    time.sleep(delay)  # Wait before retrying
                else:
                    raise
        logging.error("Failed to execute query after maximum retries due to persistent deadlocks.")
        raise RuntimeError("Query execution failed after multiple retries due to deadlock.")

    def get_current_status_document_node(self, file_name):
        query = """
                MATCH(d:Document {fileName : $file_name}) RETURN d.status AS Status , d.processingTime AS processingTime, 
                d.nodeCount AS nodeCount, d.model as model, d.relationshipCount as relationshipCount,
                d.total_chunks AS total_chunks , d.fileSize as fileSize, 
                d.is_cancelled as is_cancelled, d.processed_chunk as processed_chunk, d.fileSource as fileSource,
                d.chunkNodeCount AS chunkNodeCount,
                d.chunkRelCount AS chunkRelCount,
                d.entityNodeCount AS entityNodeCount,
                d.entityEntityRelCount AS entityEntityRelCount,
                d.communityNodeCount AS communityNodeCount,
                d.communityRelCount AS communityRelCount,
                d.createdAt AS created_time
                """
        param = {"file_name": file_name}
        return self.execute_query(query, param)

    def delete_file_from_graph(self, filenames, source_types, deleteEntities: str, merged_dir: str):

        filename_list = list(map(str.strip, json.loads(filenames)))
        source_types_list = list(map(str.strip, json.loads(source_types)))

        for (file_name, source_type) in zip(filename_list, source_types_list):
            merged_file_path = os.path.join(merged_dir, file_name)
            logging.info(f'Deleted File Path: {merged_file_path} and Deleted File Name : {file_name}')
            delete_uploaded_local_file(merged_file_path, file_name)

        query_to_delete_document = """
            MATCH (d:Document)
            WHERE d.fileName IN $filename_list AND coalesce(d.fileSource, "None") IN $source_types_list
            WITH COLLECT(d) AS documents
            CALL (documents) {
            UNWIND documents AS d
            optional match (d)<-[:PART_OF]-(c:Chunk) 
            detach delete c, d
            } IN TRANSACTIONS OF 1 ROWS
            """
        query_to_delete_document_and_entities = """
            MATCH (d:Document)
            WHERE d.fileName IN $filename_list AND coalesce(d.fileSource, "None") IN $source_types_list
            WITH COLLECT(d) AS documents
            CALL (documents) {
            UNWIND documents AS d
            OPTIONAL MATCH (d)<-[:PART_OF]-(c:Chunk)
            OPTIONAL MATCH (c:Chunk)-[:HAS_ENTITY]->(e)
            WITH d, c, e, documents
            WHERE NOT EXISTS {
                MATCH (e)<-[:HAS_ENTITY]-(c2)-[:PART_OF]->(d2:Document)
                WHERE NOT d2 IN documents
                }
            WITH d, COLLECT(c) AS chunks, COLLECT(e) AS entities
            FOREACH (chunk IN chunks | DETACH DELETE chunk)
            FOREACH (entity IN entities | DETACH DELETE entity)
            DETACH DELETE d
            } IN TRANSACTIONS OF 1 ROWS
            """
        query_to_delete_communities = """
            MATCH (c:`__Community__`)
            WHERE c.level = 0 AND NOT EXISTS { ()-[:IN_COMMUNITY]->(c) }
            DETACH DELETE c
            WITH 1 AS dummy
            UNWIND range(1, $max_level)  AS level
            CALL (level) {
                MATCH (c:`__Community__`)
                WHERE c.level = level AND NOT EXISTS { ()-[:PARENT_COMMUNITY]->(c) }
                DETACH DELETE c
                }
        """
        param = {"filename_list": filename_list, "source_types_list": source_types_list}
        community_param = {"max_level": MAX_COMMUNITY_LEVELS}
        if deleteEntities == "true":
            logging.info(f"Deleting {len(filename_list)} documents = '{filename_list}' and entities from database")
            _ = self.execute_query(query_to_delete_document_and_entities, param)
            _ = self.execute_query(query_to_delete_communities, community_param)
        else:
            logging.info(f"Deleting {len(filename_list)} documents = '{filename_list}' from database")
            _ = self.execute_query(query_to_delete_document, param)
        return len(filename_list)

    def delete_unconnected_nodes(self):
        logging.info(f"Deleting unconnected nodes from database")
        query = """
            MATCH (n)
            WHERE NOT (n)--() AND NOT n:User
            DELETE n
            """
        self.execute_query(query)

    def get_duplicate_nodes_list(self):
        duplicate_score_value = 0.97
        duplicate_text_distance = 3
        query_duplicate_nodes = """
                MATCH (n:!Chunk&!Session&!Document&!`__Community__`) with n 
                WHERE n.embedding is not null and n.id is not null // and size(toString(n.id)) > 3
                WITH n ORDER BY count {{ (n)--() }} DESC, size(toString(n.id)) DESC // updated
                WITH collect(n) as nodes
                UNWIND nodes as n
                WITH n, [other in nodes 
                // only one pair, same labels e.g. Person with Person
                WHERE elementId(n) < elementId(other) and labels(n) = labels(other)
                // at least embedding similarity of X
                AND 
                (
                // either contains each other as substrings or has a text edit distinct of less than 3
                (size(toString(other.id)) > 2 AND toLower(toString(n.id)) CONTAINS toLower(toString(other.id))) OR 
                (size(toString(n.id)) > 2 AND toLower(toString(other.id)) CONTAINS toLower(toString(n.id)))
                OR (size(toString(n.id))>5 AND apoc.text.distance(toLower(toString(n.id)), toLower(toString(other.id))) < $duplicate_text_distance)
                OR
                vector.similarity.cosine(other.embedding, n.embedding) > $duplicate_score_value
                )] as similar
                WHERE size(similar) > 0 
                // remove duplicate subsets
                with collect([n]+similar) as all
                CALL {{ with all
                    unwind all as nodes
                    with nodes, all
                    // skip current entry if it's smaller and a subset of any other entry
                    where none(other in all where other <> nodes and size(other) > size(nodes) and size(apoc.coll.subtract(nodes, other))=0)
                    return head(nodes) as n, tail(nodes) as similar
                }}
                OPTIONAL MATCH (doc:Document)<-[:PART_OF]-(c:Chunk)-[:HAS_ENTITY]->(n)
                {return_statement}
                """
        return_query_duplicate_nodes = """
                RETURN n {.*, embedding:null, elementId:elementId(n), labels:labels(n)} as e, 
                [s in similar | s {.id, .description, labels:labels(s), elementId: elementId(s)}] as similar,
                collect(distinct doc.fileName) as documents, count(distinct c) as chunkConnections
                ORDER BY e.id ASC
                LIMIT 100
                """
        total_duplicate_nodes = "RETURN COUNT(DISTINCT(n)) as total"

        param = {"duplicate_score_value": duplicate_score_value, "duplicate_text_distance": duplicate_text_distance}

        nodes_list = self.execute_query(query_duplicate_nodes.format(return_statement=return_query_duplicate_nodes),
                                        param=param)
        total_nodes = self.execute_query(query_duplicate_nodes.format(return_statement=total_duplicate_nodes),
                                         param=param)
        return nodes_list, total_nodes[0]

    def merge_duplicate_nodes(self, duplicate_nodes_list):
        nodes_list = json.loads(duplicate_nodes_list)
        logging.info(f'Nodes list to merge {nodes_list}')
        query = """
        UNWIND $rows AS row
        CALL { with row
        MATCH (first) WHERE elementId(first) = row.firstElementId
        MATCH (rest) WHERE elementId(rest) IN row.similarElementIds
        WITH first, collect (rest) as rest
        WITH [first] + rest as nodes
        CALL apoc.refactor.mergeNodes(nodes, 
        {properties:"discard",mergeRels:true, produceSelfRel:false, preserveExistingSelfRels:false, singleElementAsArray:true}) 
        YIELD node
        RETURN size(nodes) as mergedCount
        }
        RETURN sum(mergedCount) as totalMerged
        """
        param = {"rows": nodes_list}
        return self.execute_query(query, param)

    def update_node_relationship_count(self, document_name):
        logging.info("updating node and relationship count")
        label_query = """CALL db.labels"""
        community_flag = {'label': '__Community__'} in self.execute_query(label_query)
        if (not document_name) and (community_flag):
            result = self.execute_query(NODEREL_COUNT_QUERY_WITH_COMMUNITY)
        elif (not document_name) and (not community_flag):
            return []
        else:
            param = {"document_name": document_name}
            result = self.execute_query(NODEREL_COUNT_QUERY_WITHOUT_COMMUNITY, param)
        response = {}
        if result:
            for record in result:
                filename = record.get("filename", None)
                chunkNodeCount = int(record.get("chunkNodeCount", 0))
                chunkRelCount = int(record.get("chunkRelCount", 0))
                entityNodeCount = int(record.get("entityNodeCount", 0))
                entityEntityRelCount = int(record.get("entityEntityRelCount", 0))
                if (not document_name) and (community_flag):
                    communityNodeCount = int(record.get("communityNodeCount", 0))
                    communityRelCount = int(record.get("communityRelCount", 0))
                else:
                    communityNodeCount = 0
                    communityRelCount = 0
                nodeCount = int(chunkNodeCount) + int(entityNodeCount) + int(communityNodeCount)
                relationshipCount = int(chunkRelCount) + int(entityEntityRelCount) + int(communityRelCount)
                update_query = """
                MATCH (d:Document {fileName: $filename})
                SET d.chunkNodeCount = $chunkNodeCount,
                    d.chunkRelCount = $chunkRelCount,
                    d.entityNodeCount = $entityNodeCount,
                    d.entityEntityRelCount = $entityEntityRelCount,
                    d.communityNodeCount = $communityNodeCount,
                    d.communityRelCount = $communityRelCount,
                    d.nodeCount = $nodeCount,
                    d.relationshipCount = $relationshipCount
                """
                self.execute_query(update_query, {
                    "filename": filename,
                    "chunkNodeCount": chunkNodeCount,
                    "chunkRelCount": chunkRelCount,
                    "entityNodeCount": entityNodeCount,
                    "entityEntityRelCount": entityEntityRelCount,
                    "communityNodeCount": communityNodeCount,
                    "communityRelCount": communityRelCount,
                    "nodeCount": nodeCount,
                    "relationshipCount": relationshipCount
                })

                response[filename] = {"chunkNodeCount": chunkNodeCount,
                                      "chunkRelCount": chunkRelCount,
                                      "entityNodeCount": entityNodeCount,
                                      "entityEntityRelCount": entityEntityRelCount,
                                      "communityNodeCount": communityNodeCount,
                                      "communityRelCount": communityRelCount,
                                      "nodeCount": nodeCount,
                                      "relationshipCount": relationshipCount
                                      }

        return response

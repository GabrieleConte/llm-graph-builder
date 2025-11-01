import logging
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
import google.auth
from .shared.constants import ADDITIONAL_INSTRUCTIONS
from .shared.llm_graph_builder_exception import LLMGraphBuilderException
import re


def get_llm(model_env_value: str):
    """Retrieve the specified language model based on the model name."""

    try:
        if model_env_value.startswith('groq'):
            provider, model_name, base_url, api_key = model_env_value.split(",")
            llm = ChatGroq(
                api_key=api_key,
                model_name=model_name,
                base_url=base_url,
                temperature=0.1,
                )

        elif model_env_value.startswith('gemini'):
            provider, model_name, _, api_key = model_env_value.split(",")
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0,
            )

        elif model_env_value.startswith('ollama'):
            provider, model_name, base_url, _ = model_env_value.split(",")
            llm = ChatOllama(
                base_url=base_url,
                model=model_name,
                temperature=0
            )

        elif model_env_value.startswith('openai'):
            provider, model_name, api_key, _ = model_env_value.split(",")
            llm = ChatOpenAI(
                    api_key=api_key,
                    model=model_name,
                    temperature=0,
                )

        elif model_env_value.startswith('anthropic'):
            provider, model_name, api_key, _ = model_env_value.split(",")
            llm = ChatAnthropic(
                api_key=api_key,
                model=model_name,
                temperature=0,
                timeout=None
            )

        else:
            provider, model_name, api_endpoint, api_key = model_env_value.split(",")
            llm = ChatOpenAI(
                api_key=api_key,
                base_url=api_endpoint,
                model=model_name,
                temperature=0
            )

    except Exception as e:
        err = f"Error while creating LLM: {str(e)}"
        logging.error(err)
        raise Exception(err)

    logging.info(f"Model created")
    return llm, model_name


def get_llm_model_name(llm):
    """Extract name of llm model from llm object"""
    for attr in ["model_name", "model", "model_id"]:
        model_name = getattr(llm, attr, None)
        if model_name:
            return model_name.lower()
    print("Could not determine model name; defaulting to empty string")
    return ""


def get_combined_chunks(chunkId_chunkDoc_list, chunks_to_combine):
    combined_chunk_document_list = []
    combined_chunks_page_content = [
        "".join(
            document["chunk_doc"].page_content
            for document in chunkId_chunkDoc_list[i: i + chunks_to_combine]
        )
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]
    combined_chunks_ids = [
        [
            document["chunk_id"]
            for document in chunkId_chunkDoc_list[i: i + chunks_to_combine]
        ]
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]

    for i in range(len(combined_chunks_page_content)):
        combined_chunk_document_list.append(
            Document(
                page_content=combined_chunks_page_content[i],
                metadata={"combined_chunk_ids": combined_chunks_ids[i]},
            )
        )
    return combined_chunk_document_list


def get_chunk_id_as_doc_metadata(chunkId_chunkDoc_list):
    combined_chunk_document_list = [
        Document(
            page_content=document["chunk_doc"].page_content,
            metadata={"chunk_id": [document["chunk_id"]]},
        )
        for document in chunkId_chunkDoc_list
    ]
    return combined_chunk_document_list


async def get_graph_document_list(
        llm, combined_chunk_document_list, allowedNodes, allowedRelationship, additional_instructions=None
):
    if additional_instructions:
        additional_instructions = sanitize_additional_instruction(additional_instructions)
    graph_document_list = []

    if "diffbot_api_key" in dir(llm):
        llm_transformer = llm
    else:
        if "get_name" in dir(
                llm) and llm.get_name() != "ChatOpenAI" or llm.get_name() != "ChatVertexAI" or llm.get_name() != "AzureChatOpenAI":
            node_properties = False
            relationship_properties = False
        else:
            node_properties = ["description"]
            relationship_properties = ["description"]
        llm_transformer = LLMGraphTransformer(
            llm=llm,
            node_properties=node_properties,
            relationship_properties=relationship_properties,
            allowed_nodes=allowedNodes,
            allowed_relationships=allowedRelationship,
            ignore_tool_usage=True,
            additional_instructions=ADDITIONAL_INSTRUCTIONS + (
                additional_instructions if additional_instructions else "")
        )

    if isinstance(llm, DiffbotGraphTransformer):
        graph_document_list = llm_transformer.convert_to_graph_documents(combined_chunk_document_list)
    else:
        graph_document_list = await llm_transformer.aconvert_to_graph_documents(combined_chunk_document_list)
    return graph_document_list


async def get_graph_from_llm(model_env_value, chunkId_chunkDoc_list, allowedNodes, allowedRelationship, chunks_to_combine,
                             additional_instructions=None):
    try:
        llm, model_name = get_llm(model_env_value)
        logging.info(f"Using model: {model_name}")

        combined_chunk_document_list = get_combined_chunks(chunkId_chunkDoc_list, chunks_to_combine)
        logging.info(f"Combined {len(combined_chunk_document_list)} chunks")

        allowed_nodes = [node.strip() for node in allowedNodes.split(',') if node.strip()]
        logging.info(f"Allowed nodes: {allowed_nodes}")

        allowed_relationships = []
        if allowedRelationship:
            items = [item.strip() for item in allowedRelationship.split(',') if item.strip()]
            if len(items) % 3 != 0:
                raise LLMGraphBuilderException(
                    "allowedRelationship must be a multiple of 3 (source, relationship, target)")
            for i in range(0, len(items), 3):
                source, relation, target = items[i:i + 3]
                if source not in allowed_nodes or target not in allowed_nodes:
                    raise LLMGraphBuilderException(
                        f"Invalid relationship ({source}, {relation}, {target}): "
                        f"source or target not in allowedNodes"
                    )
                allowed_relationships.append((source, relation, target))
            logging.info(f"Allowed relationships: {allowed_relationships}")
        else:
            logging.info("No allowed relationships provided")

        graph_document_list = await get_graph_document_list(
            llm,
            combined_chunk_document_list,
            allowed_nodes,
            allowed_relationships,
            additional_instructions
        )
        return graph_document_list
    except Exception as e:
        logging.error(f"Error in get_graph_from_llm: {e}", exc_info=True)
        raise LLMGraphBuilderException(f"Error in getting graph from llm: {e}")


def sanitize_additional_instruction(instruction: str) -> str:
    """
   Sanitizes additional instruction by:
   - Replacing curly braces `{}` with `[]` to prevent variable interpretation.
   - Removing potential injection patterns like `os.getenv()`, `eval()`, `exec()`.
   - Stripping problematic special characters.
   - Normalizing whitespace.
   Args:
       instruction (str): Raw additional instruction input.
   Returns:
       str: Sanitized instruction safe for LLM processing.
   """
    logging.info("Sanitizing additional instructions")
    instruction = instruction.replace("{", "[").replace("}", "]")  # Convert `{}` to `[]` for safety
    # Step 2: Block dangerous function calls
    injection_patterns = [r"os\.getenv\(", r"eval\(", r"exec\(", r"subprocess\.", r"import os", r"import subprocess"]
    for pattern in injection_patterns:
        instruction = re.sub(pattern, "[BLOCKED]", instruction, flags=re.IGNORECASE)
    # Step 4: Normalize spaces
    instruction = re.sub(r'\s+', ' ', instruction).strip()
    return instruction

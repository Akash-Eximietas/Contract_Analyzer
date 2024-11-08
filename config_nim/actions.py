from typing import Optional, List
from llama_index.core import Settings
from nemoguardrails.actions import action
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import StreamingResponse
from llama_index.core import VectorStoreIndex
from llama_index.postprocessor.nvidia_rerank import NVIDIARerank
from pydantic import BaseModel, Field
# When you use from_documents, your Documents are split into chunks and parsed into Node objects
# By default, VectorStoreIndex stores everything in memory
import streamlit as st

# Strucutred Output Generation using LLAMA-INDEX and pydantic
class Output(BaseModel):
        """Output containing the response, page numbers, and confidence."""
        response: str = Field(..., description="The answer to the question in less than 100 tokens.")
        document_id: List[int] = Field(
            ...,
            description="The id of the document used to answer this question. Do not include it if the context is irrelevant."
        )
        confidence: float = Field(
            ...,
            description="Confidence value between 0-1 of the correctness of the result."
        )
        confidence_explanation: str = Field(
            ..., description="Explanation for the confidence score"
        )

if "query_engine_cache" not in st.session_state:
    # Global variable to cache the query_engine
    st.session_state.query_engine_cache = None

def init(structured_output=False):
    # Check if the query_engine is already initialized
    if st.session_state.query_engine_cache is not None:
        print('Using cached query engine')
        return st.session_state.query_engine_cache

    Settings.llm = NVIDIA(model="nvidia/nemotron-mini-4b-instruct")
    Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")
    Settings.text_splitter = SentenceSplitter(chunk_size=400)
    
    # load data
    documents = SimpleDirectoryReader("data").load_data()
    print(f'Loaded {len(documents)} documents')

    # create the retriever
    index = VectorStoreIndex.from_documents(documents)

    # get the query engine
    print("Nvidia Reranker")
    if structured_output:
        print("Processing with Structured Output")
        structured_llm= Settings.llm.as_structured_llm(output_cls=Output)
        st.session_state.query_engine_cache = index.as_query_engine(
                similarity_top_k=40,
                llm=structured_llm,
                node_postprocessors=[NVIDIARerank(top_n=4)]
            )
    else:
        st.session_state.query_engine_cache = index.as_query_engine(
                similarity_top_k=40,
                node_postprocessors=[NVIDIARerank(top_n=4)]
            )

    return st.session_state.query_engine_cache


def get_query_response(query_engine: BaseQueryEngine, query: str) -> str:
    """
    Function to query based on the query_engine and query string passed in.
    """
    response = query_engine.query(query)
    if isinstance(response, StreamingResponse):
        typed_response = response.get_response()
    else:
        typed_response = response
    response_str = typed_response.response
    if response_str is None:
        return ""
    return response_str

@action(is_system_action=True)
async def user_query(context: Optional[dict] = None):
    """
    Function to invoke the query_engine to query user message.
    """
    user_message = context.get("user_message")
    structured_output=False
    # if "structured output" in user_message:
    #     structured_output=True
    print('user_message is ', user_message)
    query_engine = init(structured_output=structured_output)
    return get_query_response(query_engine, user_message) 

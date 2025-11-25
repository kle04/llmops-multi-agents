from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import BaseMessage
from langchain.schema import HumanMessage, AIMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import Dict, List, Optional, Any, TypedDict
from pydantic import BaseModel
from config import Config
from datetime import datetime
import logging

logger = logging.getLogger(__name__)




class RAGState(TypedDict):
    # Input của người dùng
    query: str
    user_context: Dict[str, Any]

    # Xử lý input và tìm document liên quan
    query_embedding: Optional[List[float]]
    retrieved_documents: List[Dict[str, Any]]
    relevant_documents: List[Dict[str, Any]]
    
    context: str
    
    # Output
    answer: str
    sources: List[str]
    
    #Metadata
    messages: List[BaseMessage]
    step: str
    processing_time: float
    status: str
    error: Optional[str]

class RAGAgent:
    def __init__(self):
        """
            Khởi tạo RAG Agent
        """
        logger.info(f"Khởi tạo RAG Agent với LLM model: {Config.GOOGLE_LLM_MODEL} ...")
        logger.debug("Khởi tạo model với API Key ...")
        
        # Kiểm tra API Key
        self.google_api_key = str(Config.GOOGLE_API_KEY)
        if not self.google_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        elif self.google_api_key == "your_google_api_key":
            raise ValueError("Set your GEMINI_API_KEY variable in .env file")
        
        # Khởi tạo LLM với Google API Key
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=Config.GOOGLE_LLM_MODEL,
                temperature=Config.GOOGLE_LLM_TEMPERATURE,
                max_output_tokens=Config.GOOGLE_LLM_MAX_OUTPUT_TOKENS,
                google_api_key=Config.GOOGLE_API_KEY

            )
            logger.info(f"Khởi tạo model LLM {Config.GOOGLE_LLM_MODEL} thành công")
        except ChatGoogleGenerativeAIError as e:
            logger.error(f"Lỗi khi khởi tạo model LLM {Config.GOOGLE_LLM_MODEL}: {e}")
            exit(1)
        
        # Khởi tạo embedding model HuggingFace
        self._init_embedding()
        
        # Kết nối tới Qdrant DB
        self.qdrant_client = QdrantClient(url=Config.QDRANT_URL)
        logger.info(f"Kết nối tới Qdrant DB thành công")

        # Kiểm tra collection đã có chưa, nếu không thì tạo mới
        if not self.qdrant_client.collection_exists(Config.COLLECTION_NAME):
            logger.warning(f"Collection {Config.COLLECTION_NAME} is not existed, creating ...")
            self.qdrant_client.create_collection(
                collection_name=Config.COLLECTION_NAME,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
            logger.info(f"Collection {Config.COLLECTION_NAME} created")
        # Khởi tạo vector store
        self.vector_store = QdrantVectorStore(
            embedding=self.embeddings,
            client=self.qdrant_client, 
            collection_name=Config.COLLECTION_NAME
        )
        
        # Tạo workflow
        self.workflow = self._create_workflow()
        self.compiled_workflow = self.workflow.compile()
    def _init_embedding(self):
        model_kwargs = { "device": "cpu", "trust_remote_code": True }
        encode_kwargs = { 'normalize_embeddings': True, 'batch_size': 16 }
        logger.debug("Đang khởi tạo Embedding Model...")
        try:
                
            self.embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                show_progress=True
            )
            logger.info(f"Khởi tạo Embedding Model {Config.EMBEDDING_MODEL} thành công")
        except Exception as e:
            logger.exception(f"Không thể khởi tạo Embedding Model {Config.EMBEDDING_MODEL}: {e}")
            exit(1)

    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph(RAGState)

        workflow.add_node("retrieve_documents", self.retrieve_documents_node)
        workflow.add_node("filter_documents", self.filter_documents_node)
        workflow.add_node("aggregate_context", self.aggregate_context_node)
        workflow.add_node("generate_answer", self.generate_answer_node)
        workflow.add_node("error_handle", self.error_handle_node)

        # Set entry point
        workflow.set_entry_point("retrieve_documents")
        
        workflow.add_edge("retrieve_documents", "filter_documents")
        workflow.add_edge("filter_documents", "aggregate_context")
        workflow.add_edge("aggregate_context", "generate_answer")
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("error_handle", END)

        return workflow

    def retrieve_documents_node(self, state: RAGState) -> RAGState:
        
        try:
            logger.info("Bắt đầu truy xuất tài liệu cho truy vấn")
            state["step"] = "Truy xuất tài liệu"
            state["messages"] = add_messages(
                state.get("messages", []), 
                [HumanMessage(content="Đang tìm kiếm thông tin liên quan...")])
            
            # Kiểm tra vector store và embedding model
            if not self.qdrant_client or not self.embeddings:
                raise Exception("Qdrant Client hoặc Embedding Manager chưa được thiết lập")
            
            # Tạo embed cho query của người dùng
            embedded_query = self.embeddings.embed_query(state["query"])
            logger.debug("Đã tạo embedding cho query, thực hiện tìm kiếm Qdrant")
            search_results = self.qdrant_client.search(
                collection_name=Config.COLLECTION_NAME,
                query_vector=embedded_query,
                limit=Config.TOP_K_DOCUMENTS,  # Giới hạn số lượng results
                score_threshold=Config.SIMILARITY_THRESHOLD,
                with_payload=True,
                with_vectors=False
            )

            retrieved_documents = []
            
            for hit in search_results:
                result_dict = {
                    "id": str(hit.id),
                    "score": hit.score,
                    "content": hit.payload.get("content", ""),
                    "source": hit.payload.get("source", ""),
                    "chunk_index": hit.payload.get("chunk_index", 0),
                    "doc_id": hit.payload.get("doc_id", ""),
                    "section": hit.payload.get("section", "")
                }
                retrieved_documents.append(result_dict)


            state["retrieved_documents"] = retrieved_documents
            state["status"] = "document_retrieved"
            logger.info(f"Truy xuất {len(retrieved_documents)} tài liệu từ Qdrant")
            return state
        except Exception as e:
            logger.exception(f"Lỗi truy xuất tài liệu: {e}")
            state["error"] = f"Lỗi truy xuất tài liệu: {str(e)}"
            state["status"] = "error"
            return state

    def filter_documents_node(self, state: RAGState) -> RAGState:
        try:
            logger.info(f"Lọc tài liệu, tổng số đã truy xuất: {len(state.get('retrieved_documents', []))}")
            state["step"] = "Lọc document loại bỏ tài liệu điểm thấp"
            state["messages"] = add_messages(
                state.get("messages", []), 
                [HumanMessage(content="Đang lọc tài liệu ...")])
            # filter documents được retrieved, lưu vào filtered_docs
            relevant_documents = self.filter_documents(state["query"], state["retrieved_documents"])
            
            state["relevant_documents"] = relevant_documents
            state["status"] = "filtered_documents"
            logger.info(f"Số tài liệu liên quan sau lọc: {len(relevant_documents)}")
            return state
        except Exception as e:
            logger.exception(f"Lỗi filter tài liệu: {e}")
            state["error"] = f"Lỗi filter tài liệu: {str(e)}"
            state["status"] = "error"
            return state
        
    def aggregate_context_node(self, state: RAGState) -> RAGState:
        try:
            logger.info("Đang tổng hợp context từ tài liệu liên quan")
            state["step"] = "Tổng hợp tài liệu"
            state["messages"] = add_messages(
                state.get("messages", []), 
                [HumanMessage(content="Đang tổng hợp thông tin từ tài liệu...")]
            )
            # Tổng hợp context
            context = self.aggregate_context(state["relevant_documents"])
            state["context"] = context
            state["sources"] = list(set([doc["source"] for doc in state["relevant_documents"]]))
            state["status"] = "context_aggregated"
            logger.debug(f"Độ dài context tổng hợp: {len(context)} ký tự")

            
            return state
        
        except Exception as e:
            logger.exception(f"Lỗi tổng hợp context từ tài liệu: {e}")
            state["error"] = f"Lỗi tổng hợp context từ tài liệu: {str(e)}"
            state["status"] = "error"
            return state
        
    def generate_answer_node(self, state: RAGState) -> RAGState:
        try:
            logger.info("Bắt đầu tạo câu trả lời từ LLM")
            state["step"] = "Tạo câu trả lời, hoàn thành"
            state["messages"] = add_messages(
                state.get("messages", []), 
                [HumanMessage(content="Đang tạo câu trả lời ...")]
            )

            prompt = f"""
                Bạn là một chuyên gia tư vấn tâm lý học đường và sức khỏe tinh thần, có nhiệm vụ hỗ trợ học sinh, sinh viên, giáo viên và chuyên gia tâm lý. 

                Ngữ cảnh (các tài liệu tham khảo từ cơ sở dữ liệu):
                {state["context"]}

                Câu hỏi hoặc vấn đề người dùng đặt ra:
                {state["query"]}

                Yêu cầu phản hồi:
                - Giải thích rõ ràng, dễ hiểu, tránh ngôn ngữ học thuật phức tạp.
                - Cung cấp hướng dẫn cụ thể để giúp họ hiểu, đối diện và cải thiện vấn đề sức khỏe tinh thần.
                - Thể hiện sự lắng nghe, khích lệ và đồng cảm. 
                - Nếu câu hỏi có dấu hiệu khẩn cấp (liên quan đến tự hại, tự tử, bạo lực, khủng hoảng cảm xúc), hãy ưu tiên **an toàn**:
                    > "Nếu em đang trong tình trạng khủng hoảng hoặc có ý định làm hại bản thân, hãy liên hệ ngay với người thân, bạn bè hoặc chuyên gia tâm lý tại trường. Em không đơn độc và có người sẵn sàng giúp đỡ."

                3. Khi trả lời, luôn giữ thái độ nhân văn, tôn trọng và mang tính hỗ trợ. 
                - Không đưa ra chẩn đoán y khoa hay kết luận bệnh lý.
                - Nếu thiếu thông tin, hãy nói rõ rằng cần thêm dữ liệu hoặc nên tham khảo chuyên gia.

                Định dạng phản hồi:
                - Giải thích thân thiện, rõ ràng, có thể chia nhỏ từng ý.
                - Trình bày tự nhiên, gần gũi với học sinh – sinh viên Việt Nam.
                - *LƯU Ý*: KHÔNG trích dẫn nguồn hay viết bất kỳ thứ gì liên quan tới nội dung trích dẫn nguồn như: "Bạn có thể tham khảo tài liệu ...".
                - *LƯU Ý*: KHÔNG nên lúc nào cũng bắt đầu câu trả lời bằng những lời chào như "Xin chào bạn, tôi được hiểu rằng ...", "Rất vui được gặp bạn".
            """

            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)

            state["answer"] = answer
            state["status"] = "completed"
            state["processing_time"] = 0
            logger.info("Tạo câu trả lời thành công")
            state["messages"] = add_messages(
                state.get("messages", []), 
                [AIMessage(content=answer)])
            return state
        except Exception as e:
            logger.exception(f"Lỗi tạo câu trả lời: {e}")
            state["error"] = f"Lỗi tạo câu trả lời: {str(e)}"
            state["status"] = "error"
            return state
        
    def error_handle_node(self, state: RAGState) -> RAGState:
        message = f"Lỗi khi xử lý câu hỏi: {state.get('error', 'Lỗi không xác định')}"
        logger.error(message)
        state["messages"] = add_messages(
            state.get("messages", []), 
            [AIMessage(content=message)]
        )
        state["status"] = "error_handled"
        return state
        
    def invoke(self, query: str, user_context: Dict = None) -> Dict[str, Any]:
        init_state: RAGState = {
            "query": query,
            "user_context": user_context or {},
            "query_embedding": None,
            "retrieved_documents": [],
            "relevant_documents": [],
            "context": "",
            "answer": "",
            "sources": [],
            "messages": [],
            "step": "completed",
            "processing_time": 0,
            "status": "completed",
            "error": None
        }

        try:
            logger.info("Nhận yêu cầu invoke RAG Agent")
            start_time = datetime.now()
            final_state = self.compiled_workflow.invoke(init_state)
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Hoàn thành xử lý truy vấn, thời gian {processing_time:.2f}s, trạng thái: {final_state.get('status', 'unknown')}")

            return {
                "answer": final_state.get("answer", ""),
                "sources": final_state.get("sources", []),
                "relevant_documents_count": len(final_state.get("relevant_documents", [])),
                "total_retrieved_count": len(final_state.get("retrieved_documents", [])),
                "processing_time": processing_time,
                "status": final_state.get("status", "unknown")
            }
        
        except Exception as e:
            logger.exception(f"Lỗi xử lý invoke: {e}")
            return {
                "answer": f"Lỗi xử lý: {str(e)}",
                "sources": [],
                "relevant_documents_count": 0,
                "total_retrieved_count": 0,
                "processing_time": 0.0,
                "status": "error"
            }

    def filter_documents(self, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        relevant_docs = []
        logger.info(f"Filtering {len(docs)} documents với LLM grading")
        
        for doc in docs:
            # Documents đã được filter theo score ở Qdrant level
            check_prompt = f"""
            Hãy đánh giá xem đoạn văn bản sau có liên quan tới câu hỏi không?

            Câu hỏi: {query}

            Đoạn văn bản: {doc["content"][:500]}...
            
            Chỉ trả lời "YES" nếu liên quan hoặc "NO" nếu không liên quan.
            """
            
            try:
                messages = [HumanMessage(content=check_prompt)]
                response = self.llm.invoke(messages)
                grade = response.content.strip().upper()

                if grade == "YES":
                    logger.info(f"Document từ {doc['source']}, chunk {doc['chunk_index']} được chấp nhận (score: {doc['score']:.3f})")
                    relevant_docs.append(doc)
                elif grade == "NO":
                    logger.info(f"Document từ {doc['source']}, chunk {doc['chunk_index']} không được chấp nhận (score: {doc['score']:.3f})")
                else:
                    logger.warning(f"LLM trả lời không rõ ràng: '{grade}', giữ document dựa trên score")
                    relevant_docs.append(doc)
                    
            except Exception as e:
                logger.exception(f"LLM grading failed: {e}, giữ document dựa trên similarity score")
                relevant_docs.append(doc)
        
        logger.info(f"Filtering complete: {len(relevant_docs)}/{len(docs)} documents passed LLM grading")
        return relevant_docs
    
    def aggregate_context(self, docs: List[Dict[str, Any]]) -> str:
        if not docs:
            logger.warning("Không có tài liệu để tổng hợp context")
            return ""
        context_parts = []
        for doc in docs:
            source = f"Nguồn: {doc['source']}, "
            content = f"{source}\nNội dung: {doc['content']}"
            context_parts.append(content)
        
        context = "\n" + ("="*50 + "\n").join(context_parts)
        logger.debug(f"Tổng hợp context với {len(docs)} tài liệu")
        return context
    
    def health_check(self) -> Dict[str, Any]:
        try:
            if not self.llm:
                logger.error("LLM model chưa được khởi tạo")
                llm_status = "unhealthy"
            else:
                llm_status = "healthy"
            if not self.qdrant_client:
                logger.error("Qdrant Client chưa được khởi tạo")
                qdrant_status = "unhealthy"
            else:
                qdrant_status = "healthy"
            if not self.embeddings:
                logger.error("Embedding model chưa được khởi tạo")
                embedding_status = "unhealthy"
            else:
                embedding_status = "healthy"
            if not self.vector_store:
                logger.error("Vector store chưa được khởi tạo")
                vector_store_status = "unhealthy"
            else:
                vector_store_status = "healthy"
            
            return {
                "status": "healthy",
                "components": {
                    "llm": {
                        "llm_model": Config.GOOGLE_LLM_MODEL,
                        "status": llm_status
                    },
                    "qdrant_client": {
                        "status": qdrant_status
                    },
                    "embedding": {
                        "embedding_model": Config.EMBEDDING_MODEL,
                        "status": embedding_status
                    },
                    "vector_store": {
                        "status": vector_store_status
                    }
                },
                "protocol": "A2A",
                "qdrant_url": Config.QDRANT_URL,
                "collection_name": Config.COLLECTION_NAME,
                "workflow_type": "LangGraph",
                "workflow_nodes": list(self.workflow.nodes.keys())
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
        

           

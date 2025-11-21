from a2a.types import A2A
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from config import Config
from typing import Dict, Any, Optional, List
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import trim_messages, BaseMessage
import logging
import re
import json
from root_prompt import ROOT_INSTRUCTION
from a2a_client import RAGAgentA2AClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrchestratorAgent:
    def __init__(self):
        self.llm = None
        # Khởi tạo A2A Client nếu có RAG Agent được cấu hình
        self.a2a_client: Optional[RAGAgentA2AClient] = None
        self.prompt_template = None
        self._initialized = None
        
    
    def _init_llm(self):
        if not self.llm:
            logger.info("Đang khởi tạo LLM Model...")
            if Config.GOOGLE_API_KEY == "your_google_api_key":
                raise ValueError("Vui lòng thiết lập biến GOOGLE_API_KEY trong file .env")
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=Config.GOOGLE_LLM_MODEL,
                    temperature=Config.GOOGLE_LLM_TEMPERATURE,
                    max_output_tokens=Config.GOOGLE_LLM_MAX_OUTPUT_TOKENS,
                    google_api_key=Config.GOOGLE_API_KEY
                )
                logger.info(f"Khởi tạo LLM Model {Config.GOOGLE_LLM_MODEL} thành công")
            except ChatGoogleGenerativeAIError as e:
                logger.error(f"Lỗi khi khởi tạo LLM Model {Config.GOOGLE_LLM_MODEL}: {e}")
                exit(1)
    
    async def _init_a2a_client(self):
        try:
            self.a2a_client = RAGAgentA2AClient(base_url=Config.RAG_AGENT_URL)
            if not self.a2a_client._initialized:
                await self.a2a_client._initialize()
        except Exception as e:
            logger.warning(f"Lỗi khi khởi tạo A2A Client: {e}")
            self.a2a_client = None

    async def initialize(self):
        try:
            
            self._init_llm()
            await self._init_a2a_client()
            self._setup_prompt()
            self._initialized = True
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo Orchestrator Agent: {e}")
            exit(1)
        

    def _setup_prompt(self):
        input_vars = ["user_message", "chat_history"]
        self.prompt_template = PromptTemplate(
            input_variables=input_vars,
            template=f"""{ROOT_INSTRUCTION}
                [Thông tin cung cấp]
                - Lịch sử hội thoại gần đây (nếu có): 
                {{chat_history}}
                - Câu hỏi hiện tại của người dùng:
                {{user_message}}

                [Nhiệm vụ]
                1. Đánh giá xem người dùng đang hỏi mới hay đang nối tiếp ý trước dựa trên lịch sử hội thoại gần đây (lịch sử sẽ được lưu theo dạng từ cũ nhất tới mới nhất). 
                - Nếu họ nhắc lại, điều chỉnh ngữ cảnh từ cuộc trò chuyện nhưng tránh lặp nguyên văn câu trả lời cũ; giải thích thêm hoặc cung cấp góc nhìn khác.
                - Nếu câu hỏi chỉ xây dựng trên câu hỏi cũ nhưng cần thêm thông tin mới, hãy ưu tiên bổ sung nội dung liên quan.
                2. Nếu câu hỏi thuộc dạng chitchat đơn giản hoặc chỉ cần động viên, hãy trả lời trực tiếp và đảm bảo hướng tới mục tiêu hỗ trợ tinh thần.
                3. Nếu câu hỏi đòi hỏi kiến thức chuyên sâu hoặc cần trích dẫn tài liệu (ví dụ: kỹ thuật chăm sóc sức khỏe tinh thần, khuyến nghị chuyên môn), hãy chọn RAG Agent để lấy thông tin chính xác hơn.
                4. LUÔN LUÔN Trả về JSON với cấu trúc:
                {{{{
                    "selected_agent": "RAG Agent" hoặc "null"
                    "response": "Câu trả lời cuối cùng dành cho người dùng",
                    "sources": ["nguồn 1", "nguồn 2", ...] hoặc []
                }}}}

                [Lưu ý quan trọng]
                - KHÔNG bắt đầu câu trả lời bằng những lời chào như "Xin chào bạn, tôi được hiểu rằng ...", "Rất vui được gặp bạn" (Trừ khi các câu hỏi của người dùng có mục đích để chào hỏi).
                - Lịch sử hội thoại sẽ là các cặp thông tin như "type": "ai nếu như là tin nhắn của AI, human nếu như là tin nhắn của con người", "content": "Nội dung của tin nhắn đó". Hãy tổng hợp các context của lịch sử hội thoại này, và dùng nó hỗ trợ trả lời câu hỏi sắp tới
                - Khi cần gọi RAG Agent, đặt "selected_agent": "RAG Agent"
                - Tránh lặp lại nguyên văn phản hồi cũ; hãy diễn giải lại, mở rộng hoặc bổ trợ thông tin mới phù hợp ngữ cảnh.
                - Khi phát hiện tín hiệu nguy cấp (tự hại, bạo lực...), hãy khuyến khích người dùng kết nối ngay với người thân, thầy cô hoặc chuyên gia tâm lý.
                - Nếu lịch sử đã trả lời câu hỏi tương tự, hãy dựa vào ý chính để trả lời lại nhưng nhớ cập nhật/điều chỉnh (không copy nguyên văn).
            """
        )

    

         

    async def process_message(self, message: str, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        # Ensure components are initialized
        if not self._initialized:
            await self.initialize()
        formatted_messages = self.prompt_template.format(
            user_message=message,
            chat_history=history or None
        )
        try:
            result = await self.llm.ainvoke(formatted_messages)
            content = getattr(result, "content", str(result))
            try:
                match = re.search(r"\{[\s\S]*\}", content)
                if match:
                    decision = json.loads(match.group(0))
                    # Nếu chọn RAG Agent, gọi A2A server để lấy câu trả lời
                    if isinstance(decision, dict) and decision.get("selected_agent") == "RAG Agent":
                        # ensure A2A client initialized
                        if not self.a2a_client._initialized:
                            return {
                                "selected_agent": None,
                                "response": decision.get("response", ""),
                                "sources": None,
                            }
                        else:
                            rag_result = await self.a2a_client.send_message(message, stream=False)
                            return {
                                "selected_agent": "RAG Agent",
                                "response": rag_result.get("content", ""),
                                "sources": rag_result.get("sources", []),
                            }
                    elif isinstance(decision, dict) and decision.get("selected_agent") is None:
                        return {
                            "selected_agent": None,
                            "response": decision.get("response", ""),
                            "sources": None,
                        }
                    return decision
            except Exception as e:
                return {"selected_agent": None, "direct_response": "Xin lỗi, hiện tôi không thể xử lý yêu cầu.", "error": str(e)}
        except Exception as e:
            return {"selected_agent": None, "direct_response": "Xin lỗi, hiện tôi không thể xử lý yêu cầu.", "error": str(e)}
        
    async def health_check(self) -> Dict[str, Any]:
        try:
            await self.initialize()
            if not self._initialized:
                if not self.llm:
                    logger.error("LLM model chưa được khởi tạo")
                    llm_status = "unhealthy"
                               
                if not self.a2a_client:
                    logger.warning("A2A Client chưa được khởi tạo hoặc không cấu hình")
                    a2a_status = "unhealthy"
                else:
                    health = await self.a2a_client.health_check()
                    a2a_status = health
            else:
                status = "healthy"
                llm_status = "healthy"
                a2a_status = await self.a2a_client.health_check() 
            

            return {
                "status": status,
                "protocol": "A2A",
                "components": {
                    "llm": {
                        "llm_model": Config.GOOGLE_LLM_MODEL,
                        "status": llm_status
                    },
                    "a2a_client": {
                        "status": a2a_status
                    }
                }
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# if __name__ == "__main__":
#     import asyncio

#     async def _dev_test():
#         agent = OrchestratorAgent()
#         result = await agent.process_message("Xin chào")
#         print(f"Agent response: {result}")

#     asyncio.run(_dev_test())

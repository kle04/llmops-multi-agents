from a2a.types import A2A
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
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
        """Setup prompt template. For Gemini models that don't support SystemMessage, 
        we prepend system instructions to the first HumanMessage only when there's no history."""
        self.prompt_template = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{user_message}")
        ])

    

         

    def _convert_history_to_messages(self, history: Optional[List[Dict[str, str]]]) -> List[BaseMessage]:
        """Convert history dicts to LangChain BaseMessage objects."""
        if not history:
            return []
        
        messages = []
        for item in history:
            if not isinstance(item, dict):
                continue
            
            role = (item.get("type") or "").lower()
            content = item.get("content", "")
            
            if not content:
                continue
            
            if role in ("human", "user"):
                messages.append(HumanMessage(content=content))
            elif role in ("ai", "assistant"):
                messages.append(AIMessage(content=content))
        
        return messages

    def _trim_messages_if_needed(self, messages: List[BaseMessage], max_tokens: int = 4000) -> List[BaseMessage]:
        """Trim messages to fit within token limit if needed."""
        if not messages or not self.llm:
            return messages
        
        try:
            def token_counter(msgs: List[BaseMessage]) -> int:
                """Count tokens in messages using LLM's token counter."""
                if hasattr(self.llm, "get_num_tokens"):
                    return sum(self.llm.get_num_tokens(msg.content) for msg in msgs if hasattr(msg, "content"))
                return len(str(msgs)) // 4
            
            trimmed = trim_messages(
                messages,
                max_tokens=max_tokens,
                strategy="last",
                token_counter=token_counter
            )
            return trimmed
        except Exception as e:
            logger.warning(f"Failed to trim messages: {e}, using original messages")
            return messages

    def _build_system_instruction(self) -> str:
        """Build system instruction text for Gemini models that don't support SystemMessage."""
        return f"""{ROOT_INSTRUCTION}

        [Nhiệm vụ]
        1. Đánh giá xem người dùng đang hỏi mới hay đang nối tiếp ý trước dựa trên lịch sử hội thoại gần đây (nếu có).
        2. Nếu câu hỏi thuộc dạng chitchat đơn giản hoặc chỉ cần động viên, hãy trả lời trực tiếp.
        3. Nếu câu hỏi đòi hỏi kiến thức chuyên sâu về tâm lý, sức khỏe tinh thần, stress, lo âu, trầm cảm, hoặc cần trích dẫn tài liệu chuyên môn, hãy chọn RAG Agent.

        [QUAN TRỌNG - ĐỊNH DẠNG PHẢN HỒI BẮT BUỘC]
        BẠN PHẢI TRẢ LỜI BẰNG JSON VÀ CHỈ JSON, KHÔNG CÓ VĂN BẢN NÀO KHÁC TRƯỚC HOẶC SAU JSON.

        Cấu trúc JSON bắt buộc:
        {{
            "selected_agent": "RAG Agent" hoặc null,
            "response": "Câu trả lời cuối cùng dành cho người dùng",
            "sources": []
        }}

        Ví dụ cho câu hỏi về stress:
        {{"selected_agent": "RAG Agent", "response": "", "sources": []}}

        Ví dụ cho câu hỏi chitchat:
        {{"selected_agent": null, "response": "Câu trả lời chitchat", "sources": []}}

        [Lưu ý]
        - KHÔNG thêm bất kỳ văn bản nào trước hoặc sau JSON.
        - KHÔNG giải thích về JSON, chỉ trả về JSON thuần túy.
        - Khi cần RAG Agent, đặt "selected_agent": "RAG Agent" (chính xác chuỗi này).
        - Khi không cần RAG Agent, đặt "selected_agent": null.
        """

    async def process_message(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Process user message with chat history context."""
        if not self._initialized:
            await self.initialize()
        
        history_messages = self._convert_history_to_messages(history)
        history_messages = self._trim_messages_if_needed(history_messages)
        
        try:
            formatted_messages = self.prompt_template.format_messages(
                user_message=message,
                chat_history=history_messages
            )
            
            system_instruction = self._build_system_instruction()
            current_message = formatted_messages[-1]
            if isinstance(current_message, HumanMessage):
                if not history_messages:
                    current_message.content = f"{system_instruction}\n\nCâu hỏi hiện tại của người dùng: {message}"
                else:
                    current_message.content = f"{system_instruction}\n\n{message}"
            
            # Log LLM Input
            logger.info(f"🤔 Sending to LLM: {len(formatted_messages)} messages")
            # logger.debug(f"    Prompt content: {formatted_messages[-1].content[:500]}...") # internal logs

            result = await self.llm.ainvoke(formatted_messages)
            content = getattr(result, "content", str(result))
            
            # Log LLM Output
            logger.info(f"💡 LLM Response: {content[:100]}...") # Log start of response
            
            return await self._parse_and_route_decision(content, message)
            
        except Exception as e:
            logger.exception(f"Error processing message: {e}")
            return {
                "selected_agent": None,
                "response": "Xin lỗi, hiện tôi không thể xử lý yêu cầu.",
                "sources": None,
                "error": str(e)
            }

    async def _parse_and_route_decision(self, content: str, original_message: str) -> Dict[str, Any]:
        """Parse LLM response and route to appropriate agent."""
        try:
            content = content.strip()
            
            match = re.search(r"\{[\s\S]*\}", content)
            if not match:
                logger.warning(f"No JSON found in LLM response. Content: {content[:200]}")
                return {
                    "selected_agent": None,
                    "response": content,
                    "sources": None
                }
            
            json_str = match.group(0)
            decision = json.loads(json_str)
            
            if not isinstance(decision, dict):
                logger.warning(f"Parsed JSON is not a dict: {type(decision)}")
                return {
                    "selected_agent": None,
                    "response": content,
                    "sources": None
                }
            
            selected_agent = decision.get("selected_agent")
            logger.info(f"Parsed decision - selected_agent: {selected_agent}, has response: {bool(decision.get('response'))}")
            
            if selected_agent == "RAG Agent":
                logger.info("Routing to RAG Agent")
                return await self._handle_rag_agent_request(original_message, decision)
            else:
                logger.info("Handling directly with Orchestrator")
                return {
                    "selected_agent": None,
                    "response": decision.get("response", ""),
                    "sources": None
                }
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.error(f"Content that failed to parse: {content[:500]}")
            return {
                "selected_agent": None,
                "response": content,
                "sources": None,
                "error": "Failed to parse decision"
            }
        except Exception as e:
            logger.exception(f"Error parsing decision: {e}")
            return {
                "selected_agent": None,
                "response": "Xin lỗi, hiện tôi không thể xử lý yêu cầu.",
                "sources": None,
                "error": str(e)
            }

    async def _handle_rag_agent_request(self, message: str, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Handle request routing to RAG Agent."""
        if not self.a2a_client or not self.a2a_client._initialized:
            logger.warning("A2A client not initialized, falling back to direct response")
            return {
                "selected_agent": None,
                "response": decision.get("response", ""),
                "sources": None
            }
        
        try:
            rag_result = await self.a2a_client.send_message(message, stream=False)
            return {
                "selected_agent": "RAG Agent",
                "response": rag_result.get("content", ""),
                "sources": rag_result.get("sources", [])
            }
        except Exception as e:
            logger.error(f"RAG Agent request failed: {e}, falling back to direct response")
            return {
                "selected_agent": None,
                "response": decision.get("response", ""),
                "sources": None,
                "error": str(e)
            }
        
    async def generate_title(self, user_message: str, assistant_response: str) -> str:
        """Generate a short 3-5 word title for the session."""
        try:
            prompt = f"""
            Summarize the following interaction into a short, concise title (max 5 words).
            Do not use quotes.
            
            User: {user_message}
            Assistant: {assistant_response}
            
            Title:
            """
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"Failed to generate title: {e}")
            return "New Session"

    async def health_check(self) -> Dict[str, Any]:
        """Check health status of all components."""
        try:
            await self.initialize()
            
            if not self._initialized:
                status = "unhealthy"
                llm_status = "unhealthy" if not self.llm else "healthy"
                a2a_status = "unhealthy" if not self.a2a_client else await self.a2a_client.health_check()
            else:
                status = "healthy"
                llm_status = "healthy"
                a2a_status = await self.a2a_client.health_check() if self.a2a_client else "unhealthy"

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

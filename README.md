# LLMOps - Multi Agents - A2A 

## 1. Kiến trúc hệ thống

### 1.1. Kiến trúc tổng quan của hệ thống
![Alt text](images/architecture_multi_agents.png "Kiến trúc tổng quan của hệ thống")

### 1.2. Flow hoạt động của Context Retrieval
![Alt text](images/context_retrieval.png "Flow hoạt động của Context Retrieval")
Context Retrieval là thành phần quan trọng trong hệ thống, hoạt động theo flow sau:
1. **User Query** - Người dùng gửi câu hỏi đến hệ thống
2. **Text Embedding** - Câu hỏi được chuyển đổi thành vector embedding thông qua embedding service
3. **Vector Search** - Thực hiện tìm kiếm similarity trên Qdrant vector database để tìm các context liên quan
4. **Context Ranking** - Sắp xếp và lọc các context phù hợp nhất dựa trên độ tương đồng
5. **Context Return** - Trả về các context đã được ranked để làm đầu vào cho LLM


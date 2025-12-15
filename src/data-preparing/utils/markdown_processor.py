from typing import List, Dict
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pathlib import Path

class MarkdownProcessor:
    def __init__(self, embedding_manager=None):
        self.embedding_manager = embedding_manager

    def process_markdown(self, file_path: str) -> List[Dict]:
        """
        Process a markdown file and return a list of chunks with metadata.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Split by headers
            headers_to_split_on = [
                ("#", "header_1"),
                ("##", "header_2"),
                ("###", "header_3"),
            ]
            
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            md_header_splits = markdown_splitter.split_text(text)

            # Further split if chunks are too large
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                separators=["\n\n", "\n", ". ", " ", ""]
            )

            final_documents = []
            file_name = Path(file_path).name

            for chunk in md_header_splits:
                # Combine headers into a context string
                headers = [
                    chunk.metadata.get("header_1", ""),
                    chunk.metadata.get("header_2", ""),
                    chunk.metadata.get("header_3", "")
                ]
                # Filter out empty headers and join
                section_context = " > ".join([h for h in headers if h])
                
                # Split content if needed
                content_splits = text_splitter.split_text(chunk.page_content)
                
                for i, content_part in enumerate(content_splits):
                    doc = {
                        "content": content_part,
                        "source": file_name,
                        "chunk_index": i, # Note: This index is relative to the section now
                        "section": section_context,
                        "doc_id": f"{file_name}_{section_context}_{i}"
                    }
                    final_documents.append(doc)

            return final_documents

        except Exception as e:
            print(f"Error processing markdown file {file_path}: {e}")
            return []

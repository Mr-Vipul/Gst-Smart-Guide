# GST Smart Guide - Streamlit Application
The GST Smart Guide is a Streamlit-based application designed to help users query and retrieve relevant information from GST (Goods and Services Tax) guides or documents. The app supports popular document formats like DOCX, PDF, and PPTX, extracting textual information from them, and allowing users to ask questions related to the document contents. It uses pre-trained machine learning models for semantic search and generates concise, relevant answers based on user queries.

## Key features
1. Document Upload: Supports DOCX, PDF, and PPTX file formats for uploading GST guides or any other document.
2. Text Extraction: Extracts text from DOCX, PDF, and PPTX formats using specialized libraries like python-docx, PyPDF2, and python-pptx.
3. Text Chunking: Automatically splits large documents into smaller, more manageable chunks of text for easier processing.
4. Semantic Search: Utilizes the Sentence Transformer (all-mpnet-base-v2) model to compute semantic similarity between the user's question and document text.
5. Answer Generation: Employs a text-generation model (FLAN-T5) to generate concise, relevant answers based on the retrieved context from the document.
6. Interactive UI: Provides a simple, user-friendly interface for uploading documents, entering queries, and receiving answers.

## Technologies and Libraries Used
1. Streamlit: Provides the framework for creating a web-based UI for the application.
2. Sentence Transformers: Used for calculating semantic similarity between the user's question and the document text.
3. Transformers: Hugging Face's pipeline is used for text summarization and answer generation.
4. PyPDF2: Extracts text from PDF documents.
5. python-docx: Extracts text from DOCX documents.
6. python-pptx: Extracts text from PPTX presentations.
7. NumPy: For efficient data manipulation and processing.
8. Torch: Used by the Sentence Transformer model for deep learning-based semantic search.

## How to Run the Application

1.Clone the repository to your local machine:
```bash
git clone https://github.com/yourusername/gst-smart-guide.git
cd gst-smart-guide
```

2.Install the required dependencies:

```bash
pip install -r requirements.txt
```

3.Run the Streamlit application:
```
streamlit run gst_smart_guide.py
```
4.Open your web browser and navigate to the provided URL (usually http://localhost:8501/).

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## Conclusion
The GST Smart Guide is a powerful tool for querying documents using machine learning techniques, offering users the ability to efficiently find relevant information in large texts. Its simple interface and robust backend make it an ideal solution for navigating complex documents like GST guides.

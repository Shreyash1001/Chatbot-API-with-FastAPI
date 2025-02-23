# Chatbot API with FastAPI

This project implements a simple chatbot API using FastAPI. The chatbot answers questions based on user queries, context from previous conversations, and content from uploaded PDF files.

## Features:
1. **Ask Questions**: Submit questions and receive answers based on context and external search.
2. **PDF Upload**: Upload a PDF file and ask questions about its content.
3. **Memory/History**: Retains conversation history for better contextual responses.

## Requirements:
- Python 3.8+
- Redis server (optional but recommended for memory storage)
- SerpAPI key (for web search functionality)

## Setup:

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/chatbot-api.git
    cd chatbot-api
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # For macOS/Linux
    venv\Scripts\activate  # For Windows
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    - Create a `.env` file in the root directory and add your **SerpAPI key** and **Redis URL**.
    - Example:
      ```dotenv
      SERPAPI_KEY=your_serpapi_key_here
      REDIS_URL=redis://localhost:6379
      ```

5. Run the API:
    ```bash
    uvicorn app.main:app --reload
    ```

## Endpoints:

- **GET `/health`**: Check the status of the API.
- **POST `/ask`**: Ask a question. Returns an answer based on context and external search.
    - Example request:
      ```json
      {
        "question": "What is AI?",
        "user_id": "user1",
        "context": ""
      }
      ```
- **POST `/ask-pdf`**: Upload a PDF file and ask a question related to its content.
    - Example request:
      ```bash
      curl -X 'POST' -F 'question=What is in this PDF?' -F 'user_id=user1' -F 'file=@test.pdf' 'http://localhost:8000/ask-pdf'
      ```

## Testing:
Run the test cases using:
```bash
pytest

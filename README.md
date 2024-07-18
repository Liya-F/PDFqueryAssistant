# PDF Query Bot

PDF Query Bot is a RAG based Python application that extracts text from a PDF, chunks the text, embeds the chunks, stores the embeddings in a Faiss index, and generates responses to user queries using the Hugging Face GPT-2 model.

## Features

- Extracts text from PDF files
- Chunks text into manageable pieces
- Embeds text chunks using `SentenceTransformer`
- Stores embeddings in a Faiss index for efficient querying
- Generates responses to user queries using the Hugging Face API

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/pdf-query-bot.git
    cd pdf-query-bot
    ```

2. **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up your environment variables:**

    - Create a `.env` file in the root directory
    - Add your Hugging Face API key to the `.env` file:

    ```env
    HUGGING_FACE_API_KEY=your_hugging_face_api_key
    ```

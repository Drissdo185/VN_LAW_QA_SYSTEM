legal_rag/
├── README.md
├── requirements.txt
├── .env
├── .gitignore
├── docker-compose.yml
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── logging_config.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── embeddings.py
│   │   ├── retriever.py
│   │   └── reranker.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── document_processor.py
│   │   └── vectorstore.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── chat_engine.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── text_processor.py
│   │   └── helpers.py
│   └── api/
│       ├── __init__.py
│       └── main.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── embeddings/
└── tests/
├── __init__.py
├── test_retriever.py
└── test_processor.py
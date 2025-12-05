# AI Agent Development Tutorial

## Overview

This repository provides a comprehensive tutorial for developing AI agents, covering the complete spectrum from machine learning foundations to advanced AI agent systems. The tutorial takes a hands-on approach, guiding learners through practical implementations of machine learning, deep learning, natural language processing, large language models, retrieval-augmented generation, and AI agent development. Each module includes Jupyter notebooks, Python scripts, and real-world examples designed for progressive skill building.

## Project Structure

The tutorial is organized into eight main modules, each focusing on specific aspects of AI agent development:

- **01_prepare**: Initial planning and survey materials
- **02_setup**: Environment configuration and dependency management
- **03_ML**: Machine learning fundamentals with PyTorch and Keras
- **04_DL_foundation**: Deep learning core concepts including forward propagation, gradient descent, optimization, loss functions, activation functions, data augmentation, and normalization
- **05_NLP**: Natural language processing covering tokenization, embeddings, similarity measures, N-grams, BLEU scores, sentiment analysis, RNN architectures, and CLIP image-to-text models
- **06_LLM**: Large language model fine-tuning with Gemma, Llama3, BERT, and Chain-of-Thought training
- **07_RAG**: Retrieval-augmented generation using LangChain, prompt templates, LCEL, function calling, agents, database integration, and web scraping
- **08_AI_Agent**: Complete AI agent development including chatbots, Ollama integration, agent frameworks, LLM-MCP applications, and infrastructure graph RAG systems

## Installation
Before installation, please read [development environment setup manual](https://github.com/mac999/infra_ai_agent_tutorials/blob/main/02_setup/dev-env.docx).

### Prerequisites
- Python 3.9 or higher
- CUDA-compatible GPU with 8GB VRAM minimum (recommended for LLM training)
- 16GB RAM minimum
- 10GB free disk space

### Setup Instructions

1. Clone or download this repository

2. Navigate to the setup directory:
```
cd 02_setup
```

3. For PyTorch with CUDA support (optional but recommended. In case of CUDA 11.8 version):
```
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
```

4. Install required packages:
```
pip install -r requirements.txt
```

5. Install Ollama for local LLM deployment (required for 08_AI_Agent modules):
   - Follow instructions at the Ollama installation URL provided in the setup folder

6. Install Docker Desktop 
   - For Windows, you can download https://docs.docker.com/desktop/setup/install/windows-install/
   - For Mac, https://docs.docker.com/desktop/setup/install/mac-install/

7. Configure environment variables:
   - Create a .env file with your API keys (Hugging Face, OpenAI, etc.)
   - Refer to .env examples in relevant module directories

### Additional Dependencies

For specific modules, additional setup may be required:

- **Neo4j**: Install and configure Neo4j database for graph RAG (07_RAG/2_db and 08_AI_Agent/5_infra_graph_rag)
- **Docker**: Required for running Neo4j and FalkorDB containers
- **Jupyter**: Already included in requirements.txt for notebook execution

## Hands-on Content

### Machine Learning (03_ML)
- Tensor operations and calculations with PyTorch and Keras
- Model architecture design and implementation
- Practical training workflows

### Deep Learning Foundations (04_DL_foundation)
- Forward propagation mechanisms
- Gradient descent optimization techniques
- Solution optimization strategies
- Loss function implementations
- Activation function comparisons
- Data augmentation methods
- Normalization techniques
- Spiral dataset classification
- Clustering with Keras
- Neural network implementation from scratch
- Sine function training and inference

### Natural Language Processing (05_NLP)
- BPE tokenization training and usage
- Token embeddings with transformer models
- Embedding similarity calculations
- N-gram models and BLEU score evaluation
- Sentiment analysis implementations
- RNN architectures for sequence processing
- CLIP model for image-to-text conversion
- PDF and CSV text mining

### Large Language Models (06_LLM)
- Fine-tuning Gemma models with LoRA and quantization
- Llama3 fine-tuning workflows
- BERT fine-tuning for dialogue and classification
- Chain-of-Thought reasoning implementation
- Custom dataset preparation
- Training with Weights & Biases monitoring

### Retrieval-Augmented Generation (07_RAG)
- LangChain prompt template design
- Token usage optimization
- LangChain Expression Language (LCEL)
- Function calling mechanisms
- Agent development patterns
- Chain composition
- Database integration for RAG
- Web scraping for knowledge retrieval

### AI Agent Development (08_AI_Agent)
- Gradio-based chatbot interfaces
- Streamlit web applications
- PDF and web RAG chatbots
- Ollama local LLM deployment
- Multi-agent systems with ReAct patterns
- Code generation agents for BIM
- Model Context Protocol (MCP) server development
- Arduino simulation with MCP
- Weather API integration
- Calculator service implementation
- Infrastructure graph RAG with Neo4j and FalkorDB
- IFC file parsing and graph database conversion
- BIM graph agent with natural language querying

## Skills and Competencies

Upon completion of this tutorial, learners will acquire the following skills:

### Technical Skills
- Machine learning model development using PyTorch and TensorFlow
- Deep learning architecture design and optimization
- Natural language processing with transformer models
- Large language model fine-tuning and adaptation
- Retrieval-augmented generation system implementation
- AI agent framework development
- Graph database integration for knowledge retrieval
- Vector database operations with FAISS and ChromaDB
- Web application development with Gradio and Streamlit

### AI Agent Development
- Conversational AI system design
- Multi-modal agent architectures
- Tool integration and function calling
- Agent orchestration patterns
- Context management and memory systems
- Model Context Protocol server implementation
- Domain-specific agent customization

### Data Engineering
- IFC file parsing and transformation
- Graph database schema design
- Document processing and embedding generation
- Web scraping and data extraction
- Database query optimization

### MLOps and Deployment
- Model quantization and optimization
- Local LLM deployment with Ollama
- API development with FastAPI
- Monitoring with TensorBoard and Weights & Biases
- Version control for ML projects

## Target Audience

This tutorial is designed for intermediate to advanced learners with the following prerequisites:

- Basic Python programming knowledge
- Understanding of linear algebra and calculus fundamentals
- Familiarity with machine learning concepts
- Experience with Jupyter notebooks
- Basic understanding of neural networks

The content progressively increases in complexity, making it suitable for:
- Software engineers transitioning to AI development
- Data scientists expanding into AI agent systems
- Graduate students in computer science or related fields
- AI practitioners seeking practical implementation experience

## Usage

Each module contains numbered directories corresponding to tutorial sequences. Start with lower-numbered modules to build foundational knowledge before advancing to complex topics.

Navigate to specific module directories and open Jupyter notebooks for interactive learning:
```
jupyter notebook
```

For Python scripts, execute directly:
```
python script_name.py
```

Refer to individual README files in module directories for specific instructions and prerequisites.

## Author

This tutorial was developed for educational purposes in AI agent development and infrastructure intelligence systems.

## License

This project is provided for educational and research purposes. Users are responsible for complying with all applicable licenses for third-party libraries, models, and datasets used in this tutorial. Refer to individual library documentation for specific license terms.

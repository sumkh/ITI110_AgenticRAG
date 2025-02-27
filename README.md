# Agentic RAG Chatbot

The Model is deployed into a Gradio App and can be access here: 
* Using Groq API: <https://huggingface.co/spaces/Sumkh/Agentic_RAG_Groq>
* Using Huggingface pretrained model: <https://huggingface.co/spaces/Sumkh/AgenticRAG>

## Contributors

Name: **Brian Sum**  
Email: <sumkh2@gmail.com>

## Repository Structure

```bash
AgenticRAG/
├── app_azure                       # Deployment codes for Azure Platform
├── app_hf                          # Deployment codes for Huggingface Spaces (Localised Pretrained Models)
├── app_hf_groq                     # Deployment codes for Huggingface Spaces (GROQ API)                           
├── Documents/                            
│   ├── general                     # Copies of source documents added into vector store
│   └── mcq                         # Copies of MCQ source documents added into vector store
├── AgenticRAG_Project_2.ipynb      # Google Colab Notebook for Development Phase
├── LangGraph.ipynb                 # Google Colab Notebook for Pre-Production Phase  
├── Document.zip                    # Vector Databases for Google Colab Notebook 
├── Presentation.pdf                # Overview Presentation of Project
├── requirements.txt                # Python dependencies for the project
└── README.md                       # Project documentation and setup instructions

```

## Setup and Execution

### Prerequisites

- Python 3.9+ installed
- pip (Python package installer)

### Environment Setup

To set up and activate a virtual environment:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/sumkh/ITI110_AgenticRAG.git
   cd ITI110_AgenticRAG
   ```

   Use `cd` to move into the project directory where you want to create your environment (e.g., cd my_project)

2. **Create a virtual environment:**

   ```bash
   python3 -m venv yourenv
   ```

   Replace "env" with the name you want for your environment, like `yourenv`. This creates a folder named `yourenv` (or your chosen name) in your project directory.

3. **Activate the virtual environment:**

   ```bash
   source yourenv/bin/activate
   ```

   Replace `yourenv` if you used a different name.

4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

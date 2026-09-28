<div align="center">
<img src="assets/icon.png">

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)
![LangChain](https://img.shields.io/badge/LangChain-1c3c3c.svg?logo=langchain&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

*An AI agent that stands for **S**tructured **M**emory & **A**nalysis with **R**easoning and **T**ools*.

</div>

# 📝 **Description**

<div align="center">

***SMART Agent*** is a modular **AI agent** for advanced compositional reasoning via **semantic memory retrieval** and **tool integration**. 

It actually ***thinks*** through problems step-by-steps to deliver accurate and explainable answers.

Built specifically to tackle complex multi-step reasoning challenges like the ***GAIA-Level-1 dataset***.
</div>
<br>
<div align="right">

***It was built for the [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction)*** 🤗

</div>

# 🛠️ **Agent Architecture & Workflow**

<div align="center">

```mermaid
flowchart TB
    A[User Question]:::nodeStyle1 --> B[Retriever Node]:::nodeStyle2
    B --> E[Assistant Node]:::nodeStyle3
    E --> F{Need Tools?}:::decisionStyle
    F -- Yes --> I[Tools]:::nodeStyle5
    F -- No --> H[Direct Answer]:::nodeStyle4
    I --> E
    
    classDef nodeStyle1 fill:#555555,color:#ffffff,stroke:#000000,stroke-width:2px,font-weight: bold;
    classDef nodeStyle2 fill:#555555,color:#ffffff,stroke:#000000,stroke-width:2px,font-weight: bold;
    classDef nodeStyle3 fill:#555555,color:#ffffff,stroke:#000000,stroke-width:2px,font-weight: bold;
    classDef nodeStyle4 fill:#555555,color:#ffffff,stroke:#000000,stroke-width:2px,font-weight: bold;
    classDef nodeStyle5 fill:#555555,color:#ffffff,stroke:#000000,stroke-width:2px,font-weight: bold;
    classDef decisionStyle fill:#555555,color:#ffffff,stroke:#000000,stroke-width:2px,font-weight: bold;
```
</div>

# 📁 **Project Structure**

```
smart-agent/
├── .env                                                # Environment Variables
├── .gitattributes                                      # Git Attributes Configuration
├── .gitignore                                          # Git Ignore Rules
├── app.py                                              # Gradio Web Interface
├── LICENSE                                             # Project License
├── README.md                                           # You are here! ⬅️
├── requirements.txt                                    # Python Dependencies
│
├── assets/                                             # Visual Assets
│   ├── certificate-1.jpg                               # Certificate 1
│   ├── certificate-2.jpg                               # Certificate 2
│   └── icon.png                                        # App Logo / Icon
│
├── notebooks/                                          
│   └── notebook.ipynb                                  # Development Notebook & Analysis
│
└── src/                                                
    ├── agent.py                                        # Core Agent Implementation
    │
    ├── config/
    │   └── system_prompt.txt                           # Agent Instructions
    │
    ├── data/
    │   └── metadata.jsonl                              # GAIA Dataset Sample
    │   
    └── database/
        └── supabase_vector_store_schema.sql            # Database Setup
```

# 🔧 **Installation**

## 📋 Requirements

- **Python 3.8+**
- **API keys** (minimum required):

  - [Groq](https://console.groq.com/)
  - [Tavily](https://tavily.com/)
  - [Supabase](https://supabase.com/)

## ⚙️ Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/cristiangilsanz/smart-agent.git
   cd smart-agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize the vector database**
   - Create a new *Supabase* project.

4. **Set up your environment**
   ```bash
   # Edit .env with your API keys

   cp .env.example .env
   ```

5. **Populate the vector database**
   - Run `supabase_vector_store_schema.sql` in its *SQL Editor Console* to set up the vector database.

   - Populate it with question-answer embeddings from the *GAIA-Level-1* dataset.
      - You can do this by running all the notebook cells up to cell *2.1.1*.

## 🧠 Choose Your AI Brain

***SMART Agent*** supports multiple LLM providers. You can edit `build_graph()` in `agent.py` to select yours:

```python
# Option 1: Groq (Recommended - Fast & Free)
agent = BasicAgent(provider="groq")

# Option 2: Google Gemini
agent = BasicAgent(provider="google")

# Option 3: Hugging Face
agent = BasicAgent(provider="huggingface")
```
<div align="right">

***[Groq](https://groq.com/) is set by default.***

</div>

⚠️ *You must have the API key for the LLM you’re actually using.*

<br>

# 🚀 **See *SMART Agent* in Action**

## ⏱️ To Run a Quick Test:

1. **Go to the `src/agent` directory:**

```bash
cd src
```

2. **Edit `agent.py`:**

```python
if __name__ == "__main__":
    agent = BasicAgent(provider="groq")
    question = (
        "This is my question."
    )
    answer = agent(question)
    print(answer)
```

3. **Run `agent.py`:**

```bash
python agent.py
```

## 📏 To Launch the Gradio Interface:

1. **Run `app.py`:**

```bash
python app.py
```

# 🎮 **Try *SMART Agent* Online**

Want to see it in action without the setup? Try the live demo on [*Hugging Face Spaces*](https://huggingface.co/spaces)!

<div align="center">

**🔗 [Launch *SMART Agent*](https://huggingface.co/spaces/cristiangilsanz/smart-agent)**

</div>

You can query your own questions! Furthermore, the online version runs the full ***GAIA-Level-1 benchmark*** and shows real-time performance metrics too.

# 💡 **Tips for Best Results**

To get the most out of your interactions, ask clear and specific questions, and don’t hesitate to mix different types—whether math, factual, or research-based. This way, you’ll ensure more accurate and insightful answers.

**Examples of a great question:**

> - "What's 15% of 240, and is that enough to buy something that costs $35?"  
> - "Who wrote 'Pride and Prejudice' and what other books did they write?"  
> - "Summarize the latest financial news from the U.S"  
> - "Find what breakthroughs have been made in using bioacoustic monitoring to study deep-sea ecosystem health"

# 📚 **Tech Stack & Other Resources**

## 📦 Tech Stack

### 🧩 **Programming Language**
- [Python](https://docs.python.org/3/)

### 🛠️ **Agent Frameworks**
- [LangChain](https://python.langchain.com/docs/introduction/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)

### 🗣️ **Language Models**
- [ChatGroq](https://console.groq.com/docs/quickstart)
- [Google Generative AI](https://ai.google.dev/docs)
- [Hugging Face](https://huggingface.co/docs)

### 🔍 **External APIs**
- [Tavily Search](https://docs.tavily.com/)
- [Wikipedia API](https://wikipedia.readthedocs.io/)
- [ArXiv API](https://info.arxiv.org/help/api/)

### 🌐 **User Interface**
- [Gradio](https://www.gradio.app/docs)

### 💾 **Database**
- [Supabase](https://supabase.com/docs)

### 📊 **Data Processing**
- [Sentence Transformers](https://www.sbert.net/)
- [Pandas](https://pandas.pydata.org/docs/)

### ⚙️ **Configuration**
- [Python-Dotenv](https://pypi.org/project/python-dotenv/)
- [Pydantic](https://docs.pydantic.dev/)

## 🧪 Datasets & Research Papers
- [GAIA Benchmark](https://arxiv.org/abs/2311.12983)
- [Retrieval Augmented Generation (RAG)](https://arxiv.org/abs/2005.11401)
- [Tools by AI Agents](https://arxiv.org/abs/2304.08354)

# 🔬 **Research & Citations**
This project implements concepts from several research areas:

```bibtex
@article{SMARTAgent2025,
  title={SMARTAgent: Modular AI Agent for Advanced Compositional Reasoning},
  author={Cristian Gil Sanz},
  journal={AI Research},
  year={2025}
}
```

# 🎓 **Certificates**

<br>

<div align="center">
    <img width=49% src="assets/certificate-1.jpg">
    <img width=49% src="assets/certificate-2.jpg">
</div>

<br>

# 🙏 **Credits & Thanks**

Special thanks to [*Hugging Face*](https://huggingface.co/) for the incredible **[Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction)** ✨.

# 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# 📞 **Get Help & Connect**

- 💬 [Start a discussion](https://github.com/cristiangilsanz/smart-agent/discussions)
- 🐛 [Open an issue](https://github.com/cristiangilsanz/smart-agent/issues)
- 📧 cristiangilsanz@gmail.com

<div align="center">
  <br>

  **Made with 💖 for the AI Community**

  [⭐ Star this repo](https://github.com/cristiangilsanz/smart-agent) | [🍴 Fork it](https://github.com/cristiangilsanz/smart-agent/fork)

  <br>

  <a href="https://www.buymeacoffee.com/cristiangilsanz" target="_blank">
    <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 45px; width: 162px;">
  </a>

</div>

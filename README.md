# Commercial AI & GenAI


## 📖 Overview

This project sets out to form the base template for Generative AI projects, put together in early 2025. It provides the necessary building blocks to get your Generative AI project up and running in no time!

This project is built using Python and a host of leading packages and libraries (utilizes Llama-Index and accompanying tools) setup to be ultra transferable and user friendly.  

---

## 📁 Notes
There have been various source package upgrades that needs to be accounted for an the libs should be upgraded. 

## ✨ Features

- Easy installation
- Ease of use
- Extensive documentation

---

## 🚀 Installation

### Prerequisites
1. Ensure you have [Python 3.12+](https://www.python.org/downloads/release/python-3120/) installed.
2. Install package managers such as [pip](https://pip.pypa.io) or [poetry](https://python-poetry.org).
3. For local development create a virtual environment. 
4. Familiarity with LLM frameworks & Streamlit.
5. Configure the necessary environment file to kickstart your development. 

### Steps
1. Clone the repository:
   ```bash
   > git clone xxx
   > cd your-repo
2. Create virtual environment:
    ```bash
    > pip install virtualenv # If package is not already installed. 
    > python<version> -m -venv <virtual_environment_name> OR virtualenv <virtual_environment_name>
    
    # Linux & MacOS
    > source <virtual_environment_name>/bin/activate

    # Windows / Powershell 
    > <virtual_environment_name>/Scripts/Activate.ps1
    ```
3. Ensure you are in the root directory then run:
    ```bash
    pip install -r requirements.txt
    ```
4. Update the `.env `file with the necessary keys. 
5. To create a localhost of your Streamlit application:
    ```bash
    streamlit run <root_python_file>.py
    ```

## 🛠️ Usage
To use the packages at your disposal simply import them and configure according to the documentation [README](src/README.md). 

An example can be seen below:

```python 
from src.llm_embedding import LLM
from src.ingestion import Ingestion
from src.tools import Tools
from src.composable_memory import ComposableMemory
from src.agents import Agent 

# For this example we will use a OpenAI agent that has access to two tools. 
index = Ingestion(path='documents/',multiple=True).index(index_type=['vector','summary'])
tools = [
    Tools().query_engine(name='detail', description='Smaller more refined detailed information from the document',index=index['vector']), 
    Tools().query_engine(name='summary', description='The entire documents content - suitable for summary based questions.', index=index['summary']),
    Tools().duckduckgo()
]
agent = Agent(
    tools = tools,
    memory = ComposableMemory().composable_memory(),
    llm = LLM(host='AzureOpenAI')
).agent(agent='OpenAI')

agent.chat('What was the revenue generated for 2024 for entity 1244?')
> "Company ABC generated $1 000 235 for the year of 2024."
```

## 📁 Directory Structure
```text
.
├── .streamlit/                 # Streamlit configuration directory
    ├── config.toml             # Configuration file for themes and Streamlit behavior. 
├── documents/                  # Directory for local documents. 
├── src/                        # Package and source code location
    ├── .env                    # Environment file - needs to be configured/created before development begins.      
    ├── agent.py                # Agent package - available to import. 
    ├── composable_memory.py    # Memory package - available to import. 
    ├── ingestion.py            # Ingestion package - available to import.
    ├── llm_embedding.py        # llm & embedding package - available to import. 
    ├── tools.py                # Tools package - available to import. 
    ├── ...                     # All other associated abstractions
├── templates/                  # Root template directory
    ├── outputs/                # Structured output (Pydantic) custom templates. 
    ├── prompts/                # Custom prompt templates.
├── workflows/                  # Custom workflow templates.
├── main.py                     # The main application file for running your Streamlit application. 
├── requirements.txt            # Requirements file containing all packages needed to run project. 
├── DockerFile                  # DockerFile configuration for Streamlit app deployment. 
└── gitignore                   # Specify files for git to ignore.
```

## 📚 Documentation
For detailed usage instructions, or more information on the available packages refer to [README](src/README.md).

## 🤝 Contributing
Contributions are welcome! If you would like to contribute to the project template and libraries, create a pull request. 

The pull request should contain, at a minimum, the following elements:
1. Clear comment on what has been added/change to the packages/libraries.
2. An updated README.md file (following the existing structure).
3. Test/use cases and outcomes. 

## 📝 Acknowledgments
- llama-index
- llamahub
- llama-parse
- OpenAI
- AzureOpenAI

## 📬 Contact
For questions or support, reach out to:

- Name: Brendan de Villiers
- Email: devilliers.brendan@gmail.com

## 🌟 Support
If you find this project helpful, please ⭐️ this repository and share it!
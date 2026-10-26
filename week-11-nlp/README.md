# **RAG from Scratch: Search NUSMods Like It Understands You**

#### **28th October 2026, 7pm - 9pm**

Welcome to our NLP workshop! Ever asked a chatbot about a NUS module and gotten an answer that sounded right but wasn't? In this workshop, you will learn how computers make sense of text, why LLMs confidently make things up, and how to fix that with **Retrieval-Augmented Generation (RAG)**. Together, we will build a search engine over every NUS module on [NUSMods](https://nusmods.com), connect it to Gemini so that it answers your questions using real module data, and then measure how well it works!

The workshop is split into three sections, all in `workshop.ipynb`:
+ **Section 1 - Intro to NLP and LLMs:** why text is hard for computers, tokens, how an LLM writes, and where it gets things wrong
+ **Section 2 - Retrieval Engine:** embeddings, chunking, and semantic search
+ **Section 3 - Generation + Evaluation:** letting Gemini answer from the retrieved modules, and measuring how good the retrieval is

## **Requirements**
We will run the workshop locally in VS Code (or any IDE that runs Jupyter notebooks). Before the session, please install **Python `3.12.x`** ([download here](https://www.python.org/downloads/release/python-31210/); other versions may give rise to package compatibility issues) and **Git**, then set up the workshop materials:

```bash
git clone https://github.com/NUS-SDS-Workshops-AY2627/Sem-1.git
cd Sem-1/week-11-nlp

python3.12 -m venv .venv         # Windows: py -3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python --version                 # should print Python 3.12.x
pip install -r requirements.txt
```

If you already set up a Python 3.12 `.venv` for the Week 10 workshop, you still need a new one here, since the packages are different.

The virtual environment (`.venv`) keeps your workshop packages separate from your other projects, which reduces the likelihood of version conflicts. `requirements.txt` installs the following packages, which we will be using during the session:
+ `google-genai`
+ `sentence-transformers`
+ `numpy`
+ `pandas`
+ `matplotlib`
+ `requests`
+ `ipykernel` (lets VS Code run the notebook with your `.venv`)

Next, open the `week-11-nlp` folder in VS Code, then open `workshop.ipynb`. Click **Select Kernel** (top right of the notebook) → **Python Environments**, and pick `.venv`.

Please keep the `week-11-nlp` folder structure as is, since the notebook needs `rag_utils.py` and the `data` folder next to it.

## **Gemini API Key**
We will be using Gemini during the workshop, so please get your **own** free API key before the session. The free quota is counted per Google Cloud project, so a shared key would run out within minutes!
1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and sign in with a *personal* Google account.
2. Click **Create API key** and copy the key. Please **don't** set up billing, as that turns the free tier off.
3. Run the setup cells at the top of the notebook. When it asks for your key, paste it in and press Enter.

To skip pasting it every time, set `GEMINI_API_KEY` once in your terminal, then restart VS Code:
+ **macOS / Linux:** `export GEMINI_API_KEY="your-key"` (add this line to `~/.zshrc` or `~/.bashrc` to keep it)
+ **Windows:** `setx GEMINI_API_KEY "your-key"`

If everything is set up, the setup cells print the number of modules loaded and `Gemini says: ready`.

## **Datasets**
You do not need to download anything separately! The module data was fetched from NUSMods (AY 2026-2027) in advance and is already in the `data` folder:
+ `modules_slim.json`: every module offered in AY 2026-2027, with its code, title, description, workload and more
+ `embeddings.npy`: the precomputed embeddings for each module
+ `embeddings_no_title.npy`: the same embeddings without the module title, for trying out different chunk fields (see `search` in `rag_utils.py`)
+ `eval_set_v2.csv`: test questions with their relevant modules, used in Section 3

Feel free to reach out to any of your workshop guides - Nabil, Sara, Dharsh - for help if you encounter any issues. Thank you, and happy coding! 😄

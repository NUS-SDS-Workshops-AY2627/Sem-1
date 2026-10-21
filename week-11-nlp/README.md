# **RAG from Scratch: Search NUSMods Like It Understands You**

#### **28th October 2026, 7pm - 9pm**

Welcome to our NLP workshop! Ever asked a chatbot about a NUS module and gotten an answer that sounded right but wasn't? In this workshop, you will learn how computers make sense of text, why LLMs confidently make things up, and how to fix that with **Retrieval-Augmented Generation (RAG)**. Together, we will build a search engine over every NUS module on [NUSMods](https://nusmods.com), connect it to Gemini so that it answers your questions using real module data, and then measure how well it works!

The workshop is split into three sections, all in `workshop.ipynb`:
+ **Section 1 - Intro to NLP and LLMs:** why text is hard for computers, tokens, how an LLM writes, and where it gets things wrong
+ **Section 2 - Retrieval Engine:** embeddings, chunking, and semantic search
+ **Section 3 - Generation + Evaluation:** letting Gemini answer from the retrieved modules, and measuring how good the retrieval is

## **Requirements**
For the workshop, we highly encourage you to use Google Colab during our code-along sessions. This will help prevent any version or package conflicts that you may encounter when running the code locally. Simply open `workshop.ipynb` in Colab and run the first code cell, which downloads the rest of the workshop materials for you.

If you have prior coding experience and would prefer to run the code locally using your own IDE, you are more than welcome to pull the workshop materials from this repository into your local folder! We recommend using Python version `3.13.x`, as other versions may give rise to package compatibility issues. Please also ensure that you have the following packages installed, as we will be using them during the session (or simply run `pip install -r requirements.txt`):
+ `google-genai`
+ `sentence-transformers`
+ `numpy`
+ `pandas`
+ `matplotlib`
+ `requests`

We recommend creating a virtual environment (`venv`) to install these packages. This will help keep your workshop dependencies separate from those used in your other projects and reduce the likelihood of version conflicts. Please keep the `week-11-nlp` folder structure as is, since the notebook needs `rag_utils.py` and the `data` folder next to it.

## **Gemini API Key**
We will be using Gemini during the workshop, so please get your **own** free API key before the session. The free quota is counted per Google Cloud project, so a shared key would run out within minutes!
1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and sign in with a *personal* Google account.
2. Click **Create API key** and copy the key. Please **don't** set up billing, as that turns the free tier off.
3. In Colab, click the **key icon 🔑 (Secrets)** in the left sidebar, then **Add new secret**. Name it `GEMINI_API_KEY`, paste your key as the value, and switch **Notebook access** on.

If you are running locally, you can set `GEMINI_API_KEY` as an environment variable instead, or simply paste your key when the notebook asks for it.

## **Datasets**
You do not need to download anything separately! The module data was fetched from NUSMods (AY 2026-2027) in advance and is already in the `data` folder:
+ `modules_slim.json`: every module offered in AY 2026-2027, with its code, title, description, workload and more
+ `embeddings.npy`: the precomputed embeddings for each module
+ `embeddings_no_title.npy`: the same embeddings without the module title, used in Section 2
+ `eval_set_v2.csv`: test questions with their relevant modules, used in Section 3

Feel free to reach out to any of your workshop guides - Nabil, Sara, Dharsh - for help if you encounter any issues. Thank you, and happy coding! 😄

# DocuSense: Troubleshooting & Diagnostic Guide

**Project:** DocuSense — Semantic Search, Similarity Index & RAG Assistant  
**Course:** NAT820S Natural Language Processing and Applications  
**Institution:** Namibia University of Science and Technology (NUST)  

---

## 1. Quick Diagnostic Checklist

If you encounter an issue running DocuSense, check the following first:
1. Is Python 3.10 or 3.11 installed and accessible?
2. Are you in the project root directory (`nlp820sGROUPASSINMENT-main`)?
3. Does `data/` contain `bbc_articles.csv`, `bbc_chunks.csv`, and `bbc_embeddings.npy`?
4. Are required dependencies installed via `pip install -r requirements.txt`?

---

## 2. Common Setup & Runtime Issues

### 2.1 Windows PowerShell Virtual Environment Activation Error
- **Symptom**:
  ```powershell
  .\venv\Scripts\Activate.ps1 : File cannot be loaded because running scripts is disabled on this system.
  ```
- **Cause**: Windows PowerShell execution policy restricts running unsigned local scripts.
- **Fix**: Run the following command in PowerShell before activating:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
  .\venv\Scripts\Activate.ps1
  ```

---

### 2.2 `faiss-cpu` Installation or Import Failure
- **Symptom**: `ImportError: DLL load failed` or failure to compile FAISS on Windows.
- **Cause**: Some Windows environments lack the Microsoft Visual C++ Redistributable 2015–2022.
- **Fix**:
  1. DocuSense includes an automatic, high-performance **NumPy fallback** (`src/vector_index.py`). The application will run normally even without FAISS.
  2. To install FAISS cleanly on Windows:
     ```powershell
     pip install faiss-cpu==1.7.4
     ```

---

### 2.3 Port 7860 Already in Use
- **Symptom**:
  ```text
  OSError: Port 7860 is in use. Try a different port.
  ```
- **Fix**: Launch on a custom port using an environment variable or flag:
  ```bash
  # Windows PowerShell:
  $env:PORT="7865"; python app.py

  # Linux / macOS:
  PORT=7865 python app.py
  ```

---

### 2.4 Google Colab: `ModuleNotFoundError: No module named 'src'`
- **Symptom**: Running notebook cells in Google Colab fails to find `src`.
- **Cause**: Colab working directory defaults to `/content/` rather than the cloned repository directory.
- **Fix**: Ensure the very first cell in your notebook sets the working directory:
  ```python
  import os, sys
  repo_dir = "/content/nlp820sGROUPASSINMENT-main"
  if os.path.exists(repo_dir):
      os.chdir(repo_dir)
      if repo_dir not in sys.path:
          sys.path.insert(0, repo_dir)
  ```

---

### 2.5 API Key Configuration for Generative RAG
- **Symptom**: You wish to test Gemini or OpenAI, but the application uses extractive fallback.
- **Cause**: `LLM_PROVIDER` in `.env` is set to `extractive_fallback` or keys are missing.
- **Fix**:
  1. Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
  2. Edit `.env` to supply your key:
     ```ini
     LLM_PROVIDER=gemini
     GEMINI_API_KEY=AIzaSy...your_actual_key...
     ```
  3. Note: The local **Extractive Fallback** is the recommended default because it requires **zero external API keys** and operates with 100% deterministic reproducibility.

---

### 2.6 Out of Memory (OOM) during Embedding Generation
- **Symptom**: Process crashes or memory spikes when encoding thousands of passages.
- **Cause**: Embedding batch size is too large for machines with limited RAM.
- **Fix**:
  In `config.py`, reduce `EMBEDDING_BATCH_SIZE` from 64 to 32 or 16:
  ```python
  EMBEDDING_BATCH_SIZE = 32
  ```
  *(Note: DocuSense already includes precomputed embeddings in `data/bbc_embeddings.npy`, so regeneration is not needed for normal operation).*

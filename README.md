# Semantic Image Classifier

A Gradio web app that uses **DINOv2** embeddings and **cosine similarity** to find and separate near-duplicate images from a folder. It does not delete your originals — it produces downloadable ZIP files of unique and similar images, plus a visual comparison report.

## Features

- DINOv2-large embeddings via Hugging Face Transformers
- GPU acceleration on Apple Silicon (MPS), NVIDIA (CUDA), or CPU fallback
- Local folder path input for large batches (avoids browser upload limits)
- Side-by-side gallery comparing kept vs removed similar images
- Output ZIPs named after the source folder (e.g. `class (1)_unique.zip`)
- Original image filenames preserved inside ZIPs
- **Mass processing:** parent folder with multiple subfolders processed independently

## Requirements

- Python 3.10 or newer
- ~4 GB disk space for the DINOv2-large model (downloaded on first run)
- 8 GB+ RAM recommended for large image sets

---

## Installation

### macOS (Apple Silicon or Intel)

1. **Install Python 3** (if needed):

   ```bash
   # Using Homebrew
   brew install python@3.13
   ```

2. **Clone the repository:**

   ```bash
   git clone https://github.com/richardhtoomyat/Semantic-Image-Classifier-.git
   cd Semantic-Image-Classifier-
   ```

3. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. **Install dependencies:**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Run the app:**

   ```bash
   python main.py
   ```

6. Open the URL shown in the terminal (usually `http://127.0.0.1:7860`).

> **Apple Silicon tip:** The app will use MPS automatically. If you run out of memory, lower the batch size in the UI to `4` or `8`.

---

### Linux

1. **Install Python 3 and venv** (Debian/Ubuntu example):

   ```bash
   sudo apt update
   sudo apt install python3 python3-venv python3-pip
   ```

2. **Clone the repository:**

   ```bash
   git clone https://github.com/richardhtoomyat/Semantic-Image-Classifier-.git
   cd Semantic-Image-Classifier-
   ```

3. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. **Install dependencies:**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   For **NVIDIA GPU** support, install the CUDA-enabled PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/) before or instead of the default `torch` from `requirements.txt`.

5. **Run the app:**

   ```bash
   python main.py
   ```

6. Open `http://127.0.0.1:7860` in your browser.

> **Large folders:** If you hit "Too many open files", raise the limit before running:
>
> ```bash
> ulimit -n 8192
> python main.py
> ```

---

### Windows

1. **Install Python 3** from [python.org](https://www.python.org/downloads/). During setup, check **"Add Python to PATH"**.

2. **Clone the repository** (Git Bash or PowerShell):

   ```powershell
   git clone https://github.com/richardhtoomyat/Semantic-Image-Classifier-.git
   cd Semantic-Image-Classifier-
   ```

3. **Create and activate a virtual environment** (PowerShell):

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   If activation is blocked, run once:

   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

   **Command Prompt alternative:**

   ```cmd
   python -m venv .venv
   .venv\Scripts\activate.bat
   ```

4. **Install dependencies:**

   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   For **NVIDIA GPU** support, follow [pytorch.org](https://pytorch.org/get-started/locally/) to install the CUDA build of PyTorch.

5. **Run the app:**

   ```powershell
   python main.py
   ```

6. Open `http://127.0.0.1:7860` in your browser.

---

## Usage

1. Start the app with `python main.py`.
2. For **large image sets**, paste a local folder path into **"Local folder path for large batches"** and leave the upload box empty.
3. For **smaller sets**, use **"Upload image folder for smaller batches"**.
4. Adjust the similarity threshold (default 95%). Higher = stricter deduplication.
5. Click **Run DINOv2 Similarity Cleanup**.
6. Download:
   - `{folder_name}_unique.zip` — images kept as unique
   - `{folder_name}_similar.zip` — images removed as duplicates
   - CSV report and visual comparison gallery

### Example (single folder)

Local folder:

```text
/Users/you/Pictures/class (1)
```

Output ZIPs:

```text
class (1)_unique.zip
class (1)_similar.zip
```

### Mass processing (multiple subfolders)

Enter a **parent folder** that contains several image subfolders. The app auto-detects mass mode when there are **2 or more** immediate subfolders with images.

Input parent folder:

```text
/Users/you/Pictures/Class 1 Frames
```

Subfolders processed independently:

```text
class (1)
level_1_04
level_1_05
...
```

Output ZIPs are written **into the parent folder**:

```text
Class 1 Frames/class (1)_unique.zip
Class 1 Frames/class (1)_similar.zip
Class 1 Frames/level_1_04_unique.zip
Class 1 Frames/level_1_04_similar.zip
...
```

Each subfolder also gets a CSV report: `{subfolder}_report.csv`.

- Images in different subfolders are **never** compared to each other
- Memory is cleared between subfolders (garbage collection + GPU cache flush)
- Gallery previews in the UI show results from the first processed subfolder
- Failed subfolders are logged in the summary; remaining folders still run

---

## Model

The app loads `facebook/dinov2-large` from Hugging Face on first run. An internet connection is required for the initial download; later runs use the local cache.

Optional: set a Hugging Face token for faster downloads:

```bash
export HF_TOKEN=your_token_here   # macOS / Linux
set HF_TOKEN=your_token_here      # Windows CMD
$env:HF_TOKEN="your_token_here"   # Windows PowerShell
```

---

## License

MIT (or specify your preferred license).

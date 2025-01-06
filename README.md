# Government Documents

LLM-based metadata extraction for government documents

## Setup

Git and Python 3.10+ are required. Pull the repository and install python dependencies:

```bash
git clone https://github.com/scholarsportal/gov-docs.git
cd gov-docs
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with the following content:

```text
API_KEY = <your-Ollama-api-key>
```

## 1. Download Internet Archive Documents

Create or edit the csv file containing the Internet Archive download links in the `docs` folder. Then run the downloader:

```bash
python get_ia_files.py docs/govdocs_1.csv 10
```

The optional number parameter after the input csv file indicates the maximum number of documents to be downloaded.

## 2. OCR

You can use the `ocr_pdf.py` script to perform OCR on a pdf file. This script uses Tesseract to detect text in a pdf and then saves it to a text file. If your pdf files are located in the `docs` folder, the script can be executed like this:

```bash
python ocr_pdf.py docs --dpi 300 --contrast 0.5 --lang eng --debug --force
```

The *debug* flag will limit processing to 30 pages. The *dpi* setting controls the resolution of the images used for OCR. Try to match the input document resolution. The *contrast* setting is a multiplier that can be used to adjust the contrast of the image before OCR. Use higher values if some text is not detected and lower values if too much text is detected. If you are getting lots of garbage text, try lowering the contrast to 0.8 or even 0.7. The *lang* setting controls the language used for OCR. You can use multiple languages separated by a + sign, e.g., `--lang eng+fra` (default). Normally, the script will not redo the OCR if a .txt file already exists. Using the `--force` parameter will override this behaviour.

The output will be saved as text files in the `text` folder.

## 3. Extract Metadata

Ensure that all the text files to be processed are in the `text` folder and then run the script:

```bash
python process.py text --force --debug
```

The `--force` parameter will ensure that embeddings are recreated. By default existing embeddings are kept. The output will be written to a json file named `metadata.json`.

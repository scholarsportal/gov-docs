# Government Documents

LLM-based metadata extraction for government documents

## Setup

Pull the repository and install python dependencies:

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

## Usage

Put all the text files to be processed in the `input` folder and run the script:

```bash
python process.py input --force --debug
```

The `--force` parameter will ensure that embeddings are recreated. By default existing embeddings are kept. The output will be written to a json file named `metadata.json`.

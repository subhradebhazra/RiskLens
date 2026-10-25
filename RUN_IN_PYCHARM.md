# Run RiskLens v2.0 in PyCharm (Windows + Python 3.11.9)

## 1. Extract and open the correct folder

Extract the downloaded ZIP. In PyCharm, open the inner `RiskLens` folder—the one that contains `app.py`, `requirements.txt`, `pages/`, `src/`, and `tests/`.

Do not open a parent folder that only contains the ZIP, or a nested `src` folder.

## 2. Set up the interpreter

1. Go to **File → Settings → Project → Python Interpreter**.
2. Choose **Add Interpreter → Add Local Interpreter → Virtualenv**.
3. Select the installed **Python 3.11.9** executable as the base interpreter.
4. Set the environment location to `<project folder>\\.venv` and create it.
5. If an existing `.venv` was created with Python 3.12, create a fresh `.venv` using Python 3.11.9. Changing the interpreter selector does not always change the Python executable inside an existing virtual environment.

## 3. Open PyCharm Terminal and verify the folder

Choose **View → Tool Windows → Terminal**. The prompt should be in the project root. If it is not, change to the extracted folder, for example:

```powershell
cd C:\Users\<your-Windows-user>\Downloads\RiskLens_v2.0\RiskLens
```

Replace that example with the real path on your machine. Then run:

```powershell
python --version
```

Expected: `Python 3.11.9`.

If it prints a different version, fix the PyCharm project interpreter and open a new terminal before installing packages.

## 4. Install dependencies

Run one command at a time:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Keep the terminal open while packages install. Initial installation requires internet access. If pip fails, copy the full error text—especially the final 20–30 lines—and troubleshoot before continuing.

## 5. Optional environment configuration

RiskLens can open in deterministic demo mode without API keys. If you want a local environment file, run:

```powershell
Copy-Item .env.example .env
```

Add provider keys to `.env` only if required, and never share the file or commit real secrets to Git. External data workflows require internet access and may require credentials. The AI Quant Copilot offers a **Gemini / Agent 37** provider selector. Gemini needs `GEMINI_API_KEY`; Agent 37 needs both `AGENT37_API_KEY` and `AGENT37_INSTANCE_URL` for an already-running instance. Configure them on the API & Data Settings page and run the relevant connection test. Selecting Agent 37 does not automatically provision a cloud instance or pay to create one.

## 6. Verify the build

```powershell
python -m pytest -q
python -m compileall -q app.py pages src
```

The upgrade package was checked in its build environment with **32 tests passing** and `compileall` completing successfully. Your exact local outcome can differ due to Python, dependencies, or environment configuration.

## 7. Launch RiskLens

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, normally:

`http://localhost:8501`

Leave the terminal running while you use the app. Press `Ctrl+C` in that terminal to stop it.

If port 8501 is already occupied:

```powershell
python -m streamlit run app.py --server.port 8502
```

Then open `http://localhost:8502`.

## Troubleshooting

| Error | First thing to check |
|---|---|
| `No module named streamlit` | Confirm the correct `.venv` is selected, then run `python -m pip install -r requirements.txt` in the same terminal. |
| `requirements.txt` not found | Terminal is not at the project root. Change to the folder containing `requirements.txt`. |
| `No module named risklens` | Run from the project root and confirm that the `src/` directory exists. Use the launch command above. |
| Python shows 3.12 instead of 3.11.9 | Recreate the virtual environment using Python 3.11.9 and reopen the terminal. |
| API connection errors | Confirm internet access, provider credentials, plan/quota, and endpoint availability. The demo fixture is not live data. |
| Browser opens but a page errors | Copy the traceback from the PyCharm terminal. Do not assume that tests alone validate each interactive page. |

## Demo-day checklist

- Run tests once on the exact machine used for the demo.
- Open the Overview page and several key analytical pages.
- Check whether the data banner says `DEMO`, `FETCHED`, or `CACHED`; never present demo fixtures as live quotes.
- Explain that quantitative scenarios and model outputs are estimates under assumptions, not guarantees or trading advice.
- Keep API keys private and do not depend on an optional provider unless it has been tested on the demo network.

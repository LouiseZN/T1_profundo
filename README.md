# T1_profundo

MLP com diferenciação automática feita em NumPy.

Requisito: **Python 3.10 ou mais novo** instalado. Os comandos abaixo devem ser rodados na pasta raiz do projeto.
Eles criam o ambiente, instalam as dependências e abrem o notebook.

### Linux / macOS

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt && .venv/bin/python -m notebook casos_de_teste.ipynb
```

### Windows (PowerShell)

```powershell
python -m venv .venv; .venv\Scripts\python -m pip install -r requirements.txt; .venv\Scripts\python -m notebook casos_de_teste.ipynb
```

Para abrir de novo depois, rode só a última parte do comando:
`.venv/bin/python -m notebook casos_de_teste.ipynb` (Linux/macOS) ou `.venv\Scripts\python -m notebook casos_de_teste.ipynb` (Windows).

> O notebook baixa os datasets (MNIST e Boston Housing) do OpenML na primeira execução, então é preciso internet.

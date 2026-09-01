# Pipeline Preditivo de Churn — Tech Challenge Fase 1

Predição de cancelamento de clientes de telecomunicações, da análise exploratória à
API REST de inferência. 

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-34%20passing-brightgreen)](#testes)

---

## O problema

Uma operadora de telecomunicações perde clientes em ritmo acelerado, **26,5% da base
cancelou** no período observado, e só descobre a saída depois que ela acontece. Este
projeto entrega um modelo que estima a **probabilidade de cancelamento de cada
cliente**, servido por uma API REST, para que o time de retenção aja antes da decisão
de saída.

## Resultado

| | Baseline (LogReg) | **Campeão (Random Forest)** |
|---|---|---|
| F1 (validação cruzada) | 0,6283 ± 0,0236 | **0,6307 ± 0,0170** |
| F1 (teste) | 0,6136 | **0,6339** |
| ROC-AUC (teste) | 0,8417 | 0,8406 |
| Recall (teste) | 0,7834 | **0,7941** |

O modelo identifica **79% dos clientes que efetivamente cancelam**. A diferença para o
baseline linear é menor que um desvio-padrão da validação cruzada, a análise completa
desse resultado está no [Model Card](docs/model_card.md).

---

## Estrutura do repositório

```
tech-challenge-churn/
├── data/
│   ├── raw/                      # dataset bruto (baixado, não versionado)
│   └── processed/
├── docs/
│   ├── ml_canvas.md              # ML Canvas 
│   ├── model_card.md             # Model Card: métricas, vieses e limitações
│   └── roteiro_video_star.md     # roteiro do vídeo de entrega
├── models/
│   ├── champion_model.joblib     # pipeline completo serializado
│   └── model_metadata.json       # métricas, limiar, ambiente e data do treino
├── notebooks/
│   ├── 01_eda_e_baseline.ipynb   # EDA, métricas e baseline
│   └── 02_modelagem_e_comparacao.ipynb
├── reports/
│   ├── experimentos.csv          # tabela comparativa dos modelos
│   ├── analise_vies.csv          # métricas por subgrupo
│   └── figures/                  # gráficos gerados pelos notebooks
├── src/churn/
│   ├── config.py                 # caminhos, seed e schema das colunas
│   ├── data.py                   # carga e limpeza
│   ├── preprocessing.py          # ColumnTransformer
│   ├── models.py                 # catálogo de candidatos
│   ├── evaluate.py               # métricas técnicas e de negócio
│   ├── train.py                  # orquestração do treino
│   ├── predict.py                # serviço de inferência
│   ├── schemas.py                # contratos Pydantic
│   └── api/main.py               # aplicação FastAPI
├── tests/                        # suíte pytest (34 testes)
├── Dockerfile
├── Makefile
├── pyproject.toml
└── requirements.txt
```

**Separação entre experimentação e produção:** os notebooks *importam* de `src/churn/`,
nenhuma lógica é reescrita em duplicidade. O que roda no notebook é o mesmo código
que roda na API.

---

## Setup

Requer Python 3.10 ou superior.

```bash
git clone <url-do-repositorio>
cd tech-challenge-churn

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"            # ou: pip install -r requirements-dev.txt
```

### 1. Baixar o dataset

```bash
python -m churn.data --download
```

Baixa o *Telco Customer Churn* (IBM) para `data/raw/telco_customer_churn.csv`.
Alternativa: baixar manualmente do [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
e salvar com esse nome.

### 2. Treinar o modelo

```bash
python -m churn.train
```

Executa a validação cruzada dos três candidatos, elege o campeão e grava
`models/champion_model.joblib`, `models/model_metadata.json` e
`reports/experimentos.csv`. Leva cerca de 40 segundos e é determinístico (seed 42).

### 3. Subir a API

```bash
uvicorn churn.api.main:app --reload
```

Documentação interativa em **http://localhost:8000/docs**.

---

## API

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | Status do serviço e disponibilidade do modelo |
| `POST` | `/predict` | Propensão de churn de um cliente |
| `POST` | `/predict/batch` | Escoragem de até 1.000 clientes |
| `GET` | `/model/info` | Metadados, limiar e métricas do modelo em produção |
| `GET` | `/docs` | Swagger UI |

### Exemplo — verificação de saúde

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "random_forest",
  "api_version": "1.0.0"
}
```

### Exemplo — predição

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 79.85,
    "TotalCharges": 79.85
  }'
```

```json
{
  "churn_probability": 0.923291,
  "churn_prediction": 1,
  "risk_band": "alto",
  "threshold": 0.45,
  "model_name": "random_forest"
}
```

O campo `threshold` é o limiar calibrado no treino (0,45, e não o padrão 0,5) — a
justificativa está no Model Card. A resposta devolve a probabilidade bruta para que o
negócio possa aplicar seu próprio corte.

**Validação de entrada.** Os campos categóricos são tipados com `Literal` no Pydantic:
uma categoria inexistente ou um `tenure` negativo retorna **422** com mensagem
explicativa, em vez de gerar uma predição silenciosamente errada.

---

## Testes

```bash
pytest                     # 34 testes
pytest --cov=churn         # com relatório de cobertura
```

Cobrem:
- **Limpeza de dados** — conversão de `TotalCharges`, remoção de duplicatas,
  imutabilidade do DataFrame de entrada, rejeição de alvo inválido.
- **Pré-processamento** — alinhamento de colunas, saída sem nulos, tolerância a
  categorias nunca vistas em produção.
- **Modelos** — as três famílias treinam e produzem probabilidades válidas;
  reprodutibilidade com a mesma seed.
- **Métricas** — F1, ROC-AUC, valor de negócio e busca de limiar.
- **API** — `/health` responde 200, `/predict` retorna probabilidade válida,
  entradas inválidas retornam 422, perfil de alto risco pontua acima do de baixo risco.

Os testes que dependem do artefato treinado são pulados automaticamente (`skipif`)
quando o modelo ainda não existe, de modo que a suíte roda em CI limpo.

---

## Reprodutibilidade

- Seed 42 fixada em `churn.config.set_seeds()` e propagada para split, validação
  cruzada e estimadores.
- Dependências pinadas por versão mínima em `pyproject.toml` e `requirements.txt`.
- Ambiente de treino (versões de Python, scikit-learn, pandas e numpy) registrado em
  `models/model_metadata.json`.
- `data/raw/` fora do controle de versão: baixe com `make data`.
- `models/champion_model.joblib` **é versionado deliberadamente** (20 MB), para que a
  API suba imediatamente após o clone, sem exigir treino. Regenere quando quiser com
  `make train` — o resultado é idêntico, byte a byte, graças à seed fixa.

---

## Docker

```bash
docker build -t churn-api:latest .
docker run --rm -p 8000:8000 churn-api:latest
```

A imagem copia `models/champion_model.joblib`, que já vem versionado no repositório,
o build funciona logo após o clone.

---

## Atalhos do Makefile

```bash
make install   # instala as dependências de desenvolvimento
make data      # baixa o dataset
make train     # treina e salva o campeão
make test      # roda a suíte de testes
make api       # sobe a API com reload
make lint      # checa o estilo com ruff
```

---

## Documentação

- [ML Canvas](docs/ml_canvas.md) — stakeholders, decisões informadas e métricas de negócio
- [Model Card](docs/model_card.md) — performance, análise de viés e limitações
- [Roteiro do vídeo](docs/roteiro_video_star.md) — apresentação no método STAR
- [Notebook 1](notebooks/01_eda_e_baseline.ipynb) — EDA e baseline
- [Notebook 2](notebooks/02_modelagem_e_comparacao.ipynb) — modelagem e comparação

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) —
IBM Sample Data Sets. 7.043 clientes, 19 atributos, alvo binário.

## Autoria

Projeto desenvolvido individualmente.

| Nome | RM |
|---|---|
|Joao Vitor Moreira Silva | rm376097 |

**Vídeo de apresentação:** 

## Licença

MIT — código. O dataset segue a licença original da IBM Sample Data Sets.

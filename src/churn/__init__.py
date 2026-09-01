"""Pacote de predição de churn — Tech Challenge Fase 1 (POS Tech).

Módulos:
    config        — caminhos, seeds e definição de colunas
    data          — carregamento e limpeza
    preprocessing — ColumnTransformer de pré-processamento
    models        — catálogo de modelos candidatos
    evaluate      — métricas técnicas e de negócio
    train         — orquestração do treino e seleção do campeão
    predict       — serviço de inferência
    schemas       — contratos Pydantic da API
    api.main      — aplicação FastAPI
"""

__version__ = "1.0.0"

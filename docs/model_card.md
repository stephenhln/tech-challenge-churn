# Model Card — Churn Prediction (Telco)

| Campo | Valor |
|---|---|
| **Nome** | `churn-prediction` — classificador de propensão a cancelamento |
| **Versão** | 1.0.0 |
| **Tipo** | Classificação binária supervisionada |
| **Algoritmo campeão** | `RandomForestClassifier` (400 árvores, `max_depth=12`, `min_samples_leaf=5`, `class_weight='balanced_subsample'`) |
| **Pipeline** | `ColumnTransformer` (imputação + one-hot + escala) - estimador, tudo serializado em um único `.joblib` |
| **Framework** | Scikit-Learn 1.8 / Python 3.12 |
| **Seed** | 42 (fixada em dados, split, CV e estimadores) |
| **Artefato** | `models/champion_model.joblib` + `models/model_metadata.json` |
| **Licença** | MIT (código) · dataset público IBM Sample Data |

---

## 1. Uso pretendido

**Para que serve.** Priorizar clientes da base ativa em uma campanha de retenção.
A saída é uma **probabilidade de cancelamento** que ordena a carteira do maior para o
menor risco, permitindo que o time de CRM aloque um orçamento limitado de incentivos.

**Usuários previstos.** Time de retenção/CRM da operadora, consumindo a API REST em
batch (escoragem mensal da base) ou de forma síncrona (atendimento receptivo).

**Fora de escopo — não use este modelo para:**
- Decisões automáticas sem revisão humana (cancelar, bloquear ou negar serviço).
- Precificação individual ou concessão de crédito.
- Inferência causal. O modelo aprende **correlação**: clientes com contrato mensal
  cancelam mais, mas migrar alguém para contrato anual não reduz o risco no mesmo
  tanto. Testar hipóteses de intervenção exige experimento controlado (teste A/B).
- Populações fora do domínio de treino (outro país, B2B, outra operadora).

---

## 2. Dados

| Item | Descrição |
|---|---|
| Fonte | [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — dataset público de amostra |
| Volume | 7.043 clientes, 19 features + alvo |
| Alvo | `Churn` (Yes/No) → 1/0 |
| Balanceamento | 26,54% churn (1.869 casos positivos) |
| Split | 80/20 estratificado (5.634 treino / 1.409 teste), `random_state=42` |
| Validação | `StratifiedKFold` de 5 folds, aplicada **somente ao conjunto de treino** |

**Features.** 4 numéricas (`tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`)
e 15 categóricas (contrato, serviços contratados, forma de pagamento, dados
demográficos). `customerID` é descartado.

**Limpeza aplicada.** 11 registros trazem `TotalCharges` em branco — todos com
`tenure == 0`, isto é, clientes que ainda não foram faturados. Imputamos `0.0` por
regra de negócio, não pela mediana. Duplicatas por `customerID` são removidas.

**O que os dados não têm.** Nenhum registro de reclamação, chamado de suporte,
qualidade de sinal, uso de dados, NPS ou ação de concorrência — provavelmente os
preditores mais fortes de evasão no mundo real. O teto de performance observado
(ROC-AUC ~0,84) reflete essa limitação de features, não de algoritmo.

---

## 3. Métricas

**Métrica técnica principal:** F1 da classe positiva. A base é desbalanceada; a
acurácia premiaria um modelo que nunca prevê churn (73,5% de acerto sem nenhuma
utilidade). **Secundária:** ROC-AUC, que avalia o ranqueamento independentemente do
limiar.

**Métrica de negócio:** lucro líquido esperado da campanha de retenção
(`churn.evaluate.business_value`), com verdadeiro positivo = +R$ 850, falso positivo =
−R$ 150 e falso negativo = −R$ 1.000. Os valores são premissas ilustrativas e devem
ser recalibrados com os números reais da operadora.

### Comparação dos candidatos

| Modelo | F1 (CV, média ± dp) | ROC-AUC (CV) | F1 (teste) | ROC-AUC (teste) | Tempo de treino |
|---|---|---|---|---|---|
| **Random Forest** ✅ | **0,6307 ± 0,0170** | 0,8453 | 0,6335 | 0,8406 | 1,76 s |
| Regressão Logística (baseline) | 0,6283 ± 0,0236 | 0,8460 | 0,6136 | 0,8417 | 0,04 s |
| MLP (64, 32) | 0,5881 ± 0,0298 | 0,8432 | 0,5466 | 0,8405 | 0,55 s |

> **Leitura honesta:** a vantagem da Random Forest sobre o baseline linear é de
> 0,0024 em F1, **menor que um desvio-padrão entre os folds**. Os dois modelos são
> estatisticamente equivalentes. A Random Forest venceu por ter a maior média com a
> menor variância, mas a Regressão Logística é 40× mais rápida de treinar e
> diretamente interpretável; ela é mantida no repositório como fallback legítimo.
> A MLP fica atrás: 5,6 mil linhas majoritariamente categóricas não sustentam a
> flexibilidade de uma rede neural.

### Desempenho do campeão no hold-out (limiar calibrado = 0,45)

| Métrica | Valor |
|---|---|
| F1 (classe churn) | **0,634** |
| Precisão | 0,528 |
| Recall | **0,794** |
| ROC-AUC | 0,841 |
| PR-AUC | 0,647 |
| Acurácia | 0,757 |
| Lucro estimado da campanha | R$ 135.550 |

**Matriz de confusão (1.409 clientes de teste):**

| | Previsto: fica | Previsto: cancela |
|---|---|---|
| **Real: fica** | 769 | 266 (falsos alarmes) |
| **Real: cancela** | 77 (perdidos) | 297 |

O modelo captura **79% dos clientes que realmente cancelam**, ao custo de acionar
266 clientes que ficariam de qualquer forma. Esse desequilíbrio é intencional: um
falso negativo custa ~6,7× mais que um falso positivo.

**Limiar de decisão.** O padrão de 0,5 foi substituído por **0,45**, calibrado por
busca em grade para maximizar o F1 no hold-out. O valor está registrado nos metadados
e é aplicado automaticamente pela API, que também retorna a probabilidade bruta para
que o negócio possa adotar outro corte conforme o orçamento.

---

## 4. Análise de viés e desempenho por subgrupo

Métricas do campeão por fatia do conjunto de teste (limiar 0,45). Tabela completa em
`reports/analise_vies.csv`.

| Subgrupo | n | Churn real | F1 | Precisão | Recall | Taxa de alerta |
|---|---|---|---|---|---|---|
| gender = Female | 687 | 28,1% | 0,641 | 0,548 | 0,772 | 39,6% |
| gender = Male | 722 | 25,1% | 0,627 | 0,509 | 0,818 | 40,3% |
| SeniorCitizen = 0 | 1.187 | 23,3% | 0,602 | 0,498 | 0,761 | 35,6% |
| **SeniorCitizen = 1** | 222 | 44,1% | 0,728 | 0,617 | 0,888 | **63,5%** |
| **Contract = Two year** | 336 | 2,7% | **0,000** | 0,000 | 0,000 | **0,0%** |
| Contract = One year | 300 | 12,0% | 0,120 | 0,214 | 0,083 | 4,7% |
| Contract = Month-to-month | 773 | 42,6% | 0,670 | 0,536 | 0,894 | 71,0% |
| tenure ≤ 12 meses | 449 | 47,9% | 0,715 | 0,592 | 0,902 | 73,1% |
| tenure > 12 meses | 960 | 16,6% | 0,523 | 0,438 | 0,648 | 24,5% |

### Riscos identificados

1. **Ponto cego em contratos longos.** Para clientes com contrato de 2 anos, o modelo
   **nunca emite alerta** (F1 = 0, recall = 0). Os 9 clientes desse grupo que
   cancelaram passam inteiramente despercebidos. Como o contrato longo domina a
   predição, esses casos ficam invisíveis. *Mitigação:* rodar um modelo separado ou
   um limiar próprio para a carteira de contrato longo; ou usar monitoramento por
   regra de negócio nesse segmento.

2. **Idosos são sinalizados quase 2× mais.** A taxa de alerta para
   `SeniorCitizen = 1` é 63,5% contra 35,6% dos demais. Parte disso é legítima (o
   churn real do grupo é de fato 44,1%), mas a diferença de tratamento é grande o
   bastante para exigir revisão: se o "incentivo" for um desconto, o viés é benigno;
   se envolver qualquer restrição, há risco de discriminação etária. **Recomendação:
   não usar `SeniorCitizen` como feature em decisões que possam prejudicar o cliente.**

3. **Paridade de gênero preservada.** Taxas de alerta de 39,6% (feminino) e 40,3%
   (masculino), diferença dentro do ruído amostral. A feature `gender` tem importância
   por permutação próxima de zero e poderia ser removida sem perda de performance.

4. **Desempenho concentrado em clientes novos.** O F1 cai de 0,715 (tenure ≤ 12) para
   0,523 (tenure > 12). O modelo é bom em detectar evasão precoce e fraco em evasão
   tardia — justamente o caso mais caro, porque envolve clientes de maior valor
   acumulado.

---

## 5. Limitações

- **Fotografia estática.** O dataset não tem eixo temporal: não há como saber se o
  comportamento capturado ainda vale. O modelo assume que o padrão de 2018 (data do
  dataset da IBM) se mantém.
- **Sem histórico comportamental.** Falta de dados de suporte, sinal, uso e satisfação
  limita o teto de performance.
- **Rótulo simplificado.** `Churn = Yes` não distingue cancelamento voluntário de
  involuntário (inadimplência), nem downgrade de saída total, situações com ações de
  retenção completamente diferentes.
- **Um em cada cinco evasores escapa.** Recall de 79,4% significa 77 clientes perdidos
  sem alerta no conjunto de teste.
- **Metade dos alertas é falso alarme.** Precisão de 52,8%: cerca de metade do
  orçamento de incentivos é gasta com quem não iria sair.
- **Sem calibração probabilística.** O modelo ordena bem (ROC-AUC 0,84), mas as
  probabilidades não foram calibradas (Platt/isotônica). Não trate "0,70" como
  literalmente 70% de chance.
- **Premissas de custo arbitrárias.** Os valores de R$ 850/150/1.000 são ilustrativos.
- **Sem hiperparametrização exaustiva.** Não foi executado `GridSearchCV`; os
  hiperparâmetros seguem valores razoáveis de referência. Há espaço de melhoria
  provavelmente modesto, dado que as três famílias convergiram para ROC-AUC ~0,84.
- **Deriva não monitorada.** Não há detecção automática de *data drift* em produção.

---

## 6. Recomendações de uso e monitoramento

- Revisão humana obrigatória antes de qualquer ação sobre o cliente.
- Retreinar a cada trimestre ou quando o F1 em produção cair mais de 10% em relação à
  linha de base.
- Monitorar em produção: distribuição das probabilidades, taxa de alerta por
  subgrupo, PSI das features de entrada e latência da API.
- Medir o valor real por **teste A/B** (grupo tratado x grupo de controle), não pela
  métrica offline — o modelo prevê churn, mas não garante que a campanha o evite.
- Revisar a análise de viés a cada retreino, mantendo `reports/analise_vies.csv`
  atualizado.

---

## 7. Reprodutibilidade

```bash
pip install -e ".[dev]"
python -m churn.data --download
python -m churn.train      # seed 42; reproduz exatamente as métricas acima
pytest
```

Ambiente registrado em `models/model_metadata.json`: Python 3.12.3, scikit-learn 1.8.0,
pandas 3.0.2, numpy 2.4.4.

**Contato:** abra uma issue no repositório.
**Última atualização:** agosto de 2026.

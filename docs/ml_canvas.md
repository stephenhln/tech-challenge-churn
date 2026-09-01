# ML Canvas — Predição de Churn em Telecomunicações

> Preenchido na Etapa 1 (Ciclo de Vida de Projetos de ML) e revisado ao final do projeto.

---

## 1. Proposta de valor

**Qual o problema?**
A operadora perde clientes em ritmo acelerado (26,5% da base cancelou no período
observado) e só descobre o cancelamento depois que ele acontece. A retenção reativa,
agir quando o cliente liga para cancelar, tem taxa de sucesso baixa e custo alto.

**Qual o valor da solução?**
Antecipar quais clientes têm alta propensão a cancelar, permitindo que o time de CRM
aja **antes** da decisão de saída, com um orçamento de campanha limitado direcionado
a quem tem maior risco.

**Para quem?**
Time de Retenção/CRM (usuário direto), Diretoria Comercial (patrocinador) e o cliente
final (recebe uma oferta relevante em vez de contato genérico).

---

## 2. Stakeholders

| Stakeholder | Interesse | O que espera da solução |
|---|---|---|
| Diretoria Comercial | Reduzir a taxa de churn e proteger a receita | Indicador de risco confiável e ROI mensurável da campanha |
| Time de CRM / Retenção | Priorizar contatos com orçamento limitado | Lista mensal ranqueada por risco, com faixa acionável |
| TI / Engenharia | Manter o serviço no ar | API documentada, testada e conteinerizada |
| Jurídico / Compliance | Evitar discriminação e uso indevido de dados | Model Card com análise de viés e limitações explícitas |
| Cliente final | Ser tratado de forma justa | Ofertas relevantes, sem decisões automáticas prejudiciais |

---

## 3. Decisões que a predição informa

- **Quem contatar** na campanha mensal de retenção (top N por probabilidade).
- **Qual oferta** apresentar, por faixa de risco:
  - risco **alto** (p ≥ 0,60) → oferta de maior valor + contato humano;
  - risco **médio** (0,30 ≤ p < 0,60) → oferta padrão automatizada;
  - risco **baixo** (p < 0,30) → nenhuma ação (evita canibalizar receita).
- **Priorização do onboarding** dos clientes nos 12 primeiros meses, faixa em que o
  churn é mais alto e o modelo é mais preciso.

A decisão final é **sempre humana**. O modelo prioriza, não age sozinho.

---

## 4. Tarefa de ML

| Item | Definição |
|---|---|
| Tipo | Classificação binária supervisionada |
| Entrada | 19 atributos cadastrais e contratuais do cliente |
| Saída | Probabilidade de churn (0–1) + classe (0/1) + faixa de risco |
| Unidade de predição | Um cliente ativo |
| Horizonte | Propensão a cancelar no próximo ciclo (o dataset não datado impede precisar a janela) |

---

## 5. Fontes de dados

- **Treino:** IBM Telco Customer Churn (7.043 registros, público, Kaggle).
- **Produção (hipotética):** CRM (dados cadastrais e contrato), sistema de faturamento
  (`MonthlyCharges`, `TotalCharges`) e catálogo de serviços contratados.
- **Ausentes e desejáveis:** chamados de suporte, indicadores de qualidade de rede,
  consumo de dados, NPS e ofertas de concorrentes.

---

## 6. Coleta e atualização

- Escoragem em **batch mensal** de toda a base ativa (via `POST /predict/batch`).
- Escoragem **síncrona** no atendimento receptivo (via `POST /predict`).
- Retreino trimestral ou disparado por queda de performance.
- Rótulos novos vêm do próprio sistema de cancelamento, com defasagem de um ciclo.

---

## 7. Engenharia de atributos

- Imputação de `TotalCharges` ausente com 0,0 (cliente sem faturamento).
- One-hot com `handle_unknown='ignore'` para as 15 categóricas.
- Padronização das numéricas (necessária para LogReg e MLP).
- Tudo dentro de um `Pipeline` serializado, garantindo que treino e produção apliquem
  exatamente a mesma transformação.

---

## 8. Construção do modelo

- **Baseline:** Regressão Logística com `class_weight='balanced'`.
- **Candidatos:** Random Forest e MLP (64, 32).
- **Validação:** `StratifiedKFold` de 5 folds sobre o treino; hold-out de 20% usado
  uma única vez, ao final.
- **Seleção:** maior média de F1 na validação cruzada.
- **Reprodutibilidade:** seed 42 em todas as etapas.

---

## 9. Métricas

| Nível | Métrica | Alvo |
|---|---|---|
| Técnica principal | F1 da classe positiva | > 0,60 (baseline: 0,61) |
| Técnica secundária | ROC-AUC | > 0,80 |
| Operacional | Recall | ≥ 0,75 (falso negativo é o erro caro) |
| Negócio (offline) | Lucro líquido da campanha | Positivo e superior ao da regra atual |
| Negócio (online) | Redução da taxa de churn no grupo tratado vs. controle | −3 p.p. em teste A/B |
| Sistema | Latência da API (p95) | < 200 ms |

---

## 10. Avaliação de impacto e riscos

**Como saber se funcionou?** Teste A/B: metade dos clientes de alto risco recebe a
campanha, metade fica como controle. A métrica que conta é a diferença de churn
efetivo entre os grupos, a métrica offline apenas indica que o modelo é bom em
apontar o risco.

**Riscos principais**
- *Canibalização de receita:* dar desconto a quem ficaria de qualquer forma
  (precisão de 52,8% significa que metade dos alertas é falso alarme).
- *Ponto cego contratual:* clientes com contrato de 2 anos nunca são alertados.
- *Viés etário:* idosos são sinalizados quase 2× mais que os demais.
- *Deriva dos dados:* mudanças de portfólio ou de mercado invalidam o padrão aprendido.
- *Uso indevido:* aplicar o escore fora do escopo de retenção (crédito, precificação).

**Mitigações:** revisão humana obrigatória, monitoramento por subgrupo, retreino
trimestral e limitações documentadas no Model Card.

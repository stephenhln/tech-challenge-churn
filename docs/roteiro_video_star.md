# Roteiro do vídeo — método STAR (5 minutos)

> Roteiro para entrega individual, ~4min50s de fala em ritmo natural. Os tempos são cumulativos.
> Grave a tela mostrando o repositório, os notebooks e a API rodando — evite ler
> slides. Cronometre o ensaio: a falha mais comum é estourar o tempo na Action e
> cortar o Result, que é justamente a parte mais avaliada.

---

## Distribuição do tempo

| Bloco | Duração | O que aparece na tela |
|---|---|---|
| Abertura | 0:00 – 0:20 | Slide de capa com nome e RM |
| **S**ituation | 0:20 – 1:00 | Gráfico da distribuição do alvo |
| **T**ask | 1:00 – 1:30 | Estrutura do repositório |
| **A**ction | 1:30 – 3:20 | Notebooks, `src/`, tabela comparativa |
| **R**esult | 3:20 – 4:40 | API rodando ao vivo + Model Card |
| Fechamento | 4:40 – 5:00 | Limitações e próximos passos |

---

## Abertura (20s)

> "Olá, meu nome é [nome], RM [número]. Este é o Tech Challenge da Fase 1: um pipeline
> preditivo de churn para uma operadora de telecomunicações, da análise exploratória
> até uma API REST em produção. Projeto desenvolvido individualmente."

---

## Situation — o problema (40s)

*Tela: notebook 01, gráfico da distribuição do alvo.*

> "Uma operadora de telecom está perdendo clientes em ritmo acelerado. No dataset da
> IBM que usamos — 7.043 clientes, 19 atributos — **26,5% da base cancelou**. O
> problema não é só a perda: é que a empresa só descobre o cancelamento **depois** que
> ele acontece, quando a retenção já é reativa, cara e pouco eficaz.
>
> A EDA mostrou onde o risco se concentra: **contrato mensal tem 42,7% de churn contra
> 2,8% no contrato de dois anos**; cheque eletrônico como forma de pagamento chega a
> 45%; e a mediana de permanência de quem cancela é de apenas 10 meses, contra 38 de
> quem fica. Ou seja, o risco está nos primeiros meses e nos contratos sem fidelidade."

---

## Task — o objetivo (30s)

*Tela: estrutura de pastas do repositório.*

> "A tarefa foi construir o ciclo completo: EDA, definição de métricas, baseline,
> comparação de três famílias de modelos, refatoração para código produtivo, testes
> automatizados e uma API de inferência.
>
> Uma decisão importante logo no início: **acurácia foi descartada**. Com 26,5% de
> churn, um modelo que sempre responde 'não vai cancelar' acerta 73,5% — e é inútil.
> Adotei o **F1 da classe positiva** como métrica técnica, ROC-AUC como secundária, e
> criamos uma **métrica de negócio**: o lucro líquido da campanha de retenção, em que o
> falso negativo custa R$ 1.000 e o falso positivo, R$ 150."

---

## Action — as decisões técnicas (1min50s)

### Dados (25s)

*Tela: célula do notebook com as 11 linhas de `TotalCharges`.*

> "Na limpeza, o único problema real foram 11 registros com `TotalCharges` em branco.
> Em vez de imputar a mediana no automático, investiguei: **todos têm `tenure` igual a
> zero** — são clientes que ainda não foram faturados. O valor correto é zero, por
> regra de negócio. Essa decisão está numa função em `src/` e coberta por teste."

### Arquitetura (30s)

*Tela: `src/churn/models.py`.*

> "Todo o pré-processamento vive dentro de um `Pipeline` do Scikit-Learn junto com o
> estimador. Isso resolve dois problemas de uma vez: na validação cruzada, o scaler é
> reajustado dentro de cada fold, sem vazamento de dados; e em produção, a API carrega
> **um único artefato** que já sabe imputar, codificar e escalar — sem risco de o
> pré-processamento do treino divergir do da inferência."

### Modelagem (35s)

*Tela: tabela comparativa (`reports/experimentos.csv`).*

> "Comparei três famílias sob o mesmo protocolo, com validação cruzada estratificada
> de cinco folds: Regressão Logística como baseline, Random Forest e um MLP.
>
> A Random Forest venceu com F1 de 0,631, mas — e isso é importante — **a Regressão
> Logística ficou em 0,628. A diferença é menor que o desvio-padrão entre os folds.**
> Na prática, os dois são estatisticamente equivalentes. Escolhi a Random Forest
> pela maior média com a menor variância, e mantive o baseline no repositório como
> fallback legítimo. O MLP ficou atrás: com 5,6 mil linhas majoritariamente
> categóricas, não há dados que sustentem a flexibilidade de uma rede neural."

### Limiar (20s)

*Tela: gráfico da calibração do limiar.*

> "Um detalhe que fez diferença: **calibrei o limiar de decisão em 0,45**, em vez de
> aceitar o 0,5 padrão. Como o falso negativo custa quase sete vezes mais que o falso
> positivo, vale trocar precisão por recall. O limiar fica registrado nos metadados e a
> API o aplica sozinha."

---

## Result — o que funciona (1min20s)

### Métricas (25s)

*Tela: matriz de confusão.*

> "No conjunto de teste, que só foi tocado uma vez: **F1 de 0,634 e ROC-AUC de 0,841**.
> Em números de negócio: dos 374 clientes que realmente cancelaram, **o modelo capturou
> 297 — 79% deles**. O custo disso são 266 falsos alarmes, o que é aceitável quando
> perder um cliente custa quase sete vezes mais que um desconto desnecessário."

### API ao vivo (35s)

*Tela: terminal com `uvicorn` rodando, depois `/docs`.*

> "A API está no ar com FastAPI. O `/health` responde com o status e o nome do modelo
> carregado. No `/predict`, mando um cliente de contrato mensal com um mês de casa e
> fibra óptica... — **probabilidade de 0,92, faixa de risco alto**.
>
> Se eu mandar uma categoria inválida, a API retorna **422 com a mensagem de erro** em
> vez de gerar uma predição silenciosamente errada — a validação é tipada com Pydantic.
> Há também `/predict/batch` para escoragem mensal da base inteira."

### Qualidade (20s)

*Tela: `pytest` rodando.*

> "São **34 testes passando**: limpeza de dados, pré-processamento, reprodutibilidade
> com seed fixa, métricas e os endpoints da API. Todo o treino é determinístico."

---

## Fechamento — honestidade técnica (20s)

*Tela: Model Card, seção de vieses.*

> "Documentei as limitações no Model Card. Duas merecem destaque: **o modelo nunca
> emite alerta para clientes de contrato de dois anos** — um ponto cego real —, e
> **clientes idosos são sinalizados quase o dobro** dos demais, o que exige revisão
> antes de qualquer uso com impacto negativo.
>
> Como próximo passo, o valor real do modelo só se comprova em **teste A/B**: ele prevê
> churn, mas não garante que a campanha o evite. Obrigado!"

---

## Checklist de gravação

- [ ] API rodando **antes** de começar a gravar (não grave o `pip install`)
- [ ] Notebooks com as saídas já executadas e visíveis
- [ ] Terminal com fonte grande (mínimo 16pt) e tema claro
- [ ] Áudio testado — som ruim reprova mais que slide feio
- [ ] Ensaio cronometrado: **corte antes dos 5:00**, não depois
- [ ] Vídeo publicado como *não listado* no YouTube e o link no README

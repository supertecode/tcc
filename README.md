# Aprendizado de Máquina para Previsão de Demanda em E-commerce Usando Análise de Sentimento

Este repositório contém o código e os resultados do trabalho de conclusão de curso de Bacharelado em Ciência da Computação da Universidade Tecnológica Federal do Paraná (UTFPR), de autoria de Murilo Emanoel Sudario Rodrigues e orientação de Simone de Almeida.

## Resumo

Este trabalho propõe a integração da análise de sentimentos, técnica de Processamento de Linguagem Natural (PLN), com modelos de aprendizado de máquina para a previsão de demanda em comércio eletrônico. A análise de sentimentos foi utilizada para extrair a polaridade (positiva, negativa ou neutra) de avaliações de clientes, convertendo dados textuais não estruturados em variáveis estruturadas aplicáveis a modelos preditivos. A base de dados utilizada foi o *Brazilian E-Commerce Public Dataset by Olist*, submetida a etapas de tratamento, normalização, codificação de variáveis categóricas e balanceamento por meio do método SMOTE. Para a etapa de modelagem, foram aplicados três algoritmos supervisionados baseados em árvores de decisão: XGBoost, Random Forest e Extra Trees. Os experimentos indicaram que o Extra Trees apresentou o melhor desempenho, alcançando acurácia de aproximadamente 0,72, seguido pela Random Forest e, por último, o XGBoost. Os resultados evidenciaram a dificuldade comum dos modelos na classificação de sentimentos neutros, mas confirmaram a viabilidade da integração de variáveis derivadas de dados textuais em tarefas de previsão de demanda. Como contribuição, o trabalho demonstra que informações subjetivas expressas por clientes podem enriquecer modelos preditivos, ampliando a compreensão do comportamento do consumidor e oferecendo subsídios para a tomada de decisão empresarial.

## 1. Introdução

A ciência de dados é um campo interdisciplinar que intersecta áreas como: ciência da computação, estatística e matemática aplicada. Tem como objetivo, a partir da coleta de dados: tratá-los, analisá-los, relacioná-los e, a partir de modelos de inteligência artificial, os quais variam de acordo com o problema proposto e os dados disponíveis, gerar previsões que sejam suficientes para tomadas de decisões nos mais diversos setores. Uma predição baseada em metodologia aplicada pode gerar como resultado uma economia de recursos e uma solução suficientemente eficaz e otimizada (QIANG et al. 2019).

Nesse âmbito, a análise de sentimentos (também chamada de mineração de opinião) é uma técnica de processamento de linguagem natural (PLN) que identifica e classifica emoções em textos. No contexto da previsão de demanda, ela possibilita mapear informações subjetivas do consumidor fazendo com que suas satisfações, frustrações e tendências de consumo sejam indicativos mais claros dentro do modelo usado.

### 1.1 Objetivos

#### 1.1.1 Objetivo Geral

O objetivo principal do trabalho é aplicar o aprendizado de máquina para realizar a previsão de demanda em um setor específico da indústria, utilizando uma base de dados que inclua, além de dados estruturados, textos provenientes de avaliações de clientes, os quais serão processados previamente à realização da predição de demanda para gerar a variável dependente.

#### 1.1.2 Objetivos Específicos

*   Definir a base de dados para realização dos experimentos necessários para o desenvolvimento da pesquisa.
*   Realizar a análise de sentimento nos textos provenientes dos clientes que estão contidos na base de dados por meio de processamento de linguagem natural e gerar a variável classe.
*   Obter a previsão de demanda aplicando de 3 a 4 diferentes algoritmos na realização dos experimentos.
*   Fazer uma análise comparativa entre as métricas resultantes de cada algoritmo aplicado na base de dados utilizada nos experimentos, tais como Acurácia, Precisão, F1-Score e Recall.

### 1.2 Justificativa

O presente trabalho busca contribuir para o estudo das técnicas de previsão de demanda na indústria por meio da integração de análise de sentimentos a algoritmos de aprendizado de máquina. Enquanto a ciência de dados já oferece uma ampla gama de métodos para trabalhar com dados estruturados, há um espaço crescente para o desenvolvimento de abordagens que tratem dados qualitativos, especialmente no contexto de sentimentos e preferências de consumidores (COSTA et al. 2021). Com o aumento da importância das opiniões e feedbacks dos consumidores para a estratégia de negócios, a análise de sentimentos tem se mostrado um campo promissor, oferecendo insights valiosos que podem ser aplicados para melhor entender as flutuações da demanda e, com isso, otimizar a alocação de recursos (CARVALHO et al. 2023)

### 1.3 Metodologia

A metodologia deste trabalho está estruturada nas seguintes etapas:

1.  **Revisão Bibliográfica:** Pesquisa de trabalhos relacionados à análise de sentimentos e previsão de demanda.
2.  **Coleta e Tratamento de Dados:** Utilização do dataset público da Olist, com limpeza e pré-processamento dos dados.
3.  **Análise de Sentimentos:** Aplicação de técnicas de PLN para classificar as avaliações dos clientes em positivas, negativas ou neutras.
4.  **Modelagem Preditiva:** Treinamento e avaliação de modelos de machine learning (XGBoost, Random Forest e Extra Trees) para prever a demanda.
5.  **Análise de Resultados:** Comparação do desempenho dos modelos utilizando métricas como acurácia, precisão, recall e F1-score.

## 3. Desenvolvimento

### 3.1 Materiais e Ferramentas

*   **Linguagem de Programação:** Python
*   **Bibliotecas:** Pandas, Scikit-learn, XGBoost, Matplotlib, Seaborn, NLTK, Transformers.
*   **Base de Dados:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

### 3.2 Métodos

O desenvolvimento do projeto seguiu as seguintes etapas:

1.  **Tratamento da Base de Dados:** Limpeza de dados nulos, remoção de outliers e transformação de variáveis categóricas.
2.  **Processamento de Linguagem Natural:** Utilização de um modelo pré-treinado para análise de sentimentos, com a criação de uma nova variável `sentiment`.
3.  **Análise Exploratória dos Dados:** Visualização da distribuição dos dados e da correlação entre as variáveis.
4.  **Normalização e Balanceamento:** Aplicação da técnica SMOTE para balancear as classes da variável alvo.
5.  **Aplicação da Previsão de Demanda:** Treinamento e avaliação dos modelos XGBoost, Random Forest e Extra Trees.

## 4. Resultados

As métricas de avaliação dos modelos foram as seguintes:

| Modelo | Acurácia | Precisão | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| XGBoost | 0.69 | 0.68 | 0.69 | 0.68 |
| Random Forest | 0.71 | 0.70 | 0.71 | 0.71 |
| Extra Trees | 0.72 | 0.71 | 0.72 | 0.72 |

As matrizes de confusão e a importância das features para cada modelo também foram analisadas, revelando que o modelo Extra Trees obteve o melhor desempenho geral.

## 5. Conclusão

### 5.1 Considerações Finais

O trabalho demonstrou a viabilidade de integrar a análise de sentimentos em modelos de previsão de demanda, enriquecendo a análise com dados subjetivos dos consumidores. O modelo Extra Trees se destacou, alcançando a maior acurácia entre os algoritmos testados.

### 5.2 Contribuições do Trabalho

*   Demonstração de que informações subjetivas expressas por clientes podem enriquecer modelos preditivos.
*   Ampliação da compreensão do comportamento do consumidor.
*   Fornecimento de subsídios para a tomada de decisão empresarial.

### 5.3 Limitações Identificadas

*   Dificuldade dos modelos em classificar sentimentos neutros.
*   Acurácia geral dos modelos pode ser aprimorada.

### 5.4 Trabalhos Futuros

*   Exploração de outros modelos de análise de sentimentos e de previsão de demanda.
*   Utilização de outras fontes de dados, como redes sociais.
*   Aprofundamento da análise das features para entender melhor os fatores que influenciam a demanda.

## Apêndice A - Dicionário de Dados

O dicionário de dados completo pode ser encontrado no apêndice do trabalho original.


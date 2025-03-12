# Análise de Sentimentos de Comentários do Reddit com Python

Este projeto realiza uma análise de sentimentos de comentários extraídos de um subreddit específico do Reddit, utilizando técnicas de Processamento de Linguagem Natural (PNL) e Machine Learning.

## Visão Geral

O script Python neste repositório faz o seguinte:

1.  **Coleta de Dados:**
    *   Utiliza a biblioteca `praw` para interagir com a API do Reddit e extrair comentários de um subreddit definido pelo usuário.
    *   É necessário configurar as credenciais da API do Reddit (Client ID, Client Secret, User Agent).

2.  **Análise de Sentimentos (VADER):**
    *   Emprega o `SentimentIntensityAnalyzer` da biblioteca `nltk.sentiment.vader` para classificar os comentários em três categorias: Positivo, Negativo e Neutro.  O VADER é um analisador de sentimentos baseado em léxico, especialmente adequado para textos de mídia social.
    *   Essa classificação inicial é usada para rotular os dados para o treinamento do modelo de Machine Learning.

3.  **Pré-processamento de Texto:**
    *   Remove pontuações e caracteres especiais.
    *   Converte o texto para minúsculas.
    *   Remove *stopwords* (palavras comuns como "a", "o", "de", etc.) em inglês.

4.  **Machine Learning (Regressão Logística):**
    *   **Vetorização:** Transforma o texto em uma representação numérica usando `TfidfVectorizer` do scikit-learn. O TF-IDF (Term Frequency-Inverse Document Frequency) atribui pesos às palavras com base em sua frequência no comentário e em todo o corpus de comentários, destacando palavras mais relevantes.  O código utiliza unigramas (palavras individuais) e bigramas (pares de palavras).
    *   **Treinamento do Modelo:** Treina um modelo de Regressão Logística usando o `LogisticRegression` do scikit-learn. O modelo aprende a relação entre as palavras (representadas pelos seus valores TF-IDF) e o sentimento (positivo ou negativo) atribuído pelo VADER.
    *   **Extração de Pesos:** Obtém os coeficientes do modelo treinado, que representam o "peso" ou a importância de cada palavra para a classificação do sentimento. Palavras com pesos positivos altos são fortes indicadores de sentimento positivo, e palavras com pesos negativos altos são fortes indicadores de sentimento negativo.
    * Divide os dados em treino e teste, para uma validação mais rigorosa.

5.  **Visualizações:**
    *   **Distribuição de Sentimentos (Gráfico de Barras):** Mostra a contagem de comentários classificados como Positivo, Negativo e Neutro.
    *   **Distribuição de Sentimentos (Gráfico de Pizza):** Apresenta a proporção de cada sentimento em relação ao total de comentários.
    *   **Top 10 Palavras Positivas/Negativas (Gráfico de Barras Horizontais):** Exibe as 10 palavras com os maiores pesos positivos e as 10 palavras com os maiores pesos negativos, de acordo com o modelo de Regressão Logística.  Isso fornece insights sobre quais palavras são mais influentes na determinação do sentimento.
    *   **Relação entre Polaridade e Subjetividade (Gráfico de Dispersão):** Explora a relação entre a polaridade (quão positivo ou negativo é o comentário) e a subjetividade (a diferença entre as pontuações positivas e negativas do VADER) dos comentários.
    *   **Evolução dos Sentimentos ao Longo do Tempo (Gráfico de Linha):** Apresenta a variação do número de comentários positivos, negativos e neutros ao longo de um período (dados de datas gerados aleatoriamente no código atual).
    *   **Nuvem de Palavras:** Cria uma representação visual das palavras mais frequentes nos comentários, onde o tamanho da palavra é proporcional à sua frequência.
    *   **Análise de Emoções (VADER):**  Calcula as médias dos scores de emoções (negativo, neutro, positivo, compound) fornecidos pelo VADER.

6.  **Salvamento dos Resultados:**
    *   Salva os resultados da análise (comentários, sentimentos, polaridade, subjetividade, etc.) em arquivos CSV e Excel.

## Requisitos

*   Python 3.6+
*   Bibliotecas Python:
    *   `praw` (para acessar a API do Reddit)
    *   `nltk` (para análise de sentimentos e pré-processamento de texto)
    *   `scikit-learn` (para Machine Learning, especificamente TF-IDF e Regressão Logística)
    *   `matplotlib` (para visualizações)
    *   `seaborn` (para visualizações)
    *   `pandas` (para manipulação de dados)
    *   `wordcloud` (para gerar a nuvem de palavras)

## Instalação

1.  **Clone este repositório:**

    ```bash
    git clone <URL_do_seu_repositório>
    cd <nome_da_pasta_do_repositório>
    ```

2.  **Crie um ambiente virtual:**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    .venv\Scripts\activate  # Windows
    ```

3.  **Instale as dependências:**

    ```bash
    pip install -r requirements.txt
    ```
    

4. **Configure as credenciais da API do Reddit**
   Crie uma aplicação no Reddit e insira suas credenciais nas variáveis `client_id`, `client_secret` e `user_agent`.

5.  **Execute o script:**

    ```bash
    python main.py
    ```

## Configuração

*   **Credenciais do Reddit:** Você precisará criar uma aplicação no Reddit ([https://www.reddit.com/prefs/apps/](https://www.reddit.com/prefs/apps/)) para obter as credenciais da API (Client ID, Client Secret, User Agent). Substitua os valores no código pelas suas credenciais.
*   **Subreddit:** Modifique a variável `subreddit_nome` no código para analisar um subreddit diferente.
* **Limite de comentários:** Altere o valor do parâmetro `limite` na função `coletar_comentarios()` para coletar mais ou menos comentários.
* **Arquivo de Stopwords:** O código carrega a lista de stopwords em inglês do NLTK (`stopwords.words('english')`).

## Saída

O script gera os seguintes arquivos:

*   `analise_sentimentos_reddit.csv`: Arquivo CSV com os resultados da análise.
*   `analise_sentimentos_reddit.xlsx`: Arquivo Excel com os resultados da análise.

Além disso, exibe vários gráficos na tela.

## Contribuições

Contribuições são bem-vindas! Se você encontrar bugs, tiver sugestões de melhorias ou quiser adicionar novas funcionalidades, sinta-se à vontade para abrir um *issue* ou enviar um *pull request*.

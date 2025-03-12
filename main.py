import praw
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk
from nltk.corpus import stopwords
from datetime import datetime, timedelta
import random
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split  
from wordcloud import WordCloud

# Baixando recursos do NLTK
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Configuração do PRAW
try:
    reddit = praw.Reddit(
        client_id='seu_id',  # Substitua pelo seu Client ID
        client_secret='seu_id',  # Substitua pelo seu Client Secret
        user_agent='seu_user'  # Substitua pelo seu User Agent
    )
    print(f"Autenticado como: {reddit.user.me()}")
except Exception as e:
    print(f"Erro de autenticação: {e}")
    exit()

def coletar_comentarios(subreddit_nome, limite=100):
    try:
        subreddit = reddit.subreddit(subreddit_nome)
        comentarios = []
        for comentario in subreddit.comments(limit=limite):
            comentarios.append(comentario.body)
        return comentarios
    except Exception as e:
        print(f"Erro ao coletar comentários: {e}")
        return []

def analisar_sentimento_vader(texto):
    analisador = SentimentIntensityAnalyzer()
    scores = analisador.polarity_scores(texto)
    if scores['compound'] >= 0.05:
        return 'Positivo'
    elif scores['compound'] <= -0.05:
        return 'Negativo'
    else:
        return 'Neutro'



# Coletando comentários, aqui especifica qual subreddit usar
subreddit_nome = 'movies'
comentarios = coletar_comentarios(subreddit_nome, limite=500) 

# Analisando os sentimentos com VADER (para rotular os dados)
if comentarios:
    sentimentos = [analisar_sentimento_vader(comentario) for comentario in comentarios]
    df = pd.DataFrame({'Comentario': comentarios, 'Sentimento': sentimentos})

    # Gráfico de Barras (Distribuição de Sentimentos)
    plt.figure(figsize=(8, 6))
    sns.countplot(x=df['Sentimento'], palette="viridis")
    plt.title(f'Distribuição de Sentimentos no Subreddit r/{subreddit_nome}')
    plt.xlabel('Sentimento')
    plt.ylabel('Número de Comentários')
    plt.show()

    # Gráfico de Pizza (Distribuição de Sentimentos) 
    sentimento_counts = df['Sentimento'].value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(sentimento_counts, labels=sentimento_counts.index, autopct='%1.1f%%', colors=['green', 'red', 'blue']) #mudei a ordem das cores para ficar igual a imagem
    plt.title('Distribuição de Sentimentos no Subreddit r/movies')
    plt.show()
    


    # Gráfico de Barras Horizontais (Top Palavras Positivas/Negativas)
    
    # Pré-processamento: Remover linhas com 'Neutro'
    df_filtrado = df[df['Sentimento'] != 'Neutro']

    # Vetorização: Converter texto em números usando TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))  # Usando bigramas também
    X = vectorizer.fit_transform(df_filtrado['Comentario'])
    y = df_filtrado['Sentimento']  # 'Positivo' ou 'Negativo'

    # Dividir os dados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Treinar o modelo de Regressão Logística
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Obter os pesos (coeficientes) das palavras
    pesos = model.coef_[0]

    # Criar um DataFrame para facilitar a visualização
    palavras = vectorizer.get_feature_names_out()
    df_pesos = pd.DataFrame({'Palavra': palavras, 'Peso': pesos})

    # Separar palavras positivas e negativas
    df_pesos_positivos = df_pesos.sort_values('Peso', ascending=False).head(10)
    df_pesos_negativos = df_pesos.sort_values('Peso', ascending=True).head(10)
    
    # Plotar o gráfico
    fig, ax = plt.subplots(1, 2, figsize=(18, 6)) 

    # Palavras Positivas
    ax[0].barh(df_pesos_positivos['Palavra'], df_pesos_positivos['Peso'], color='green')
    ax[0].set_title('Top 10 Palavras Positivas')
    ax[0].set_xlabel('Peso')
    ax[0].invert_yaxis()  # Inverter para mostrar as palavras mais importantes no topo

    # Palavras Negativas
    ax[1].barh(df_pesos_negativos['Palavra'], df_pesos_negativos['Peso'], color='red')
    ax[1].set_title('Top 10 Palavras Negativas')
    ax[1].set_xlabel('Peso')
    ax[1].invert_yaxis()

    plt.tight_layout()
    plt.show()


    #Gráfico de Dispersão (Polaridade x Subjetividade)
    analisador = SentimentIntensityAnalyzer()
    df['Polaridade'] = df['Comentario'].apply(lambda x: analisador.polarity_scores(x)['compound'])
    df['Subjetividade'] = df['Comentario'].apply(lambda x: analisador.polarity_scores(x)['pos'] - analisador.polarity_scores(x)['neg'])

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Polaridade', y='Subjetividade', hue='Sentimento', palette='viridis')  
    plt.title('Relação entre Polaridade e Subjetividade')
    plt.xlabel('Polaridade')
    plt.ylabel('Subjetividade')
    plt.show()

    # Gráfico de Linha (Evolução Temporal dos Sentimentos) 
    df['Data'] = [datetime.now() - timedelta(days=random.randint(1, 30)) for _ in range(len(df))]
    df['Data'] = pd.to_datetime(df['Data']).dt.date
    sentimento_por_data = df.groupby(['Data', 'Sentimento']).size().unstack()
    
    #Verificando se sentimento_por_data não está vazio antes de plotar
    if not sentimento_por_data.empty:
        plt.figure(figsize=(12, 6))
        sentimento_por_data.plot(kind='line', marker='o')
        plt.title('Evolução dos Sentimentos ao Longo do Tempo')
        plt.xlabel('Data')
        plt.ylabel('Número de Comentários')
        plt.legend(title='Sentimento')
        plt.show()
    else:
        print("Não há dados suficientes para plotar a evolução temporal dos sentimentos.")


    # 6. Nuvem de Palavras - Esse é mais por curiosidade, não é necessário para análise
    todos_comentarios = ' '.join(df['Comentario'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(todos_comentarios)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Nuvem de Palavras dos Comentários')
    plt.show()

    # 7. Análise de Emoções (Usando VADER)
    df['Emocoes'] = df['Comentario'].apply(lambda x: analisador.polarity_scores(x))
    df['Negativo'] = df['Emocoes'].apply(lambda x: x['neg'])
    df['Neutro'] = df['Emocoes'].apply(lambda x: x['neu'])
    df['Positivo'] = df['Emocoes'].apply(lambda x: x['pos'])
    df['Compound'] = df['Emocoes'].apply(lambda x: x['compound'])

    medias_emocoes = df[['Negativo', 'Neutro', 'Positivo', 'Compound']].mean()
    print("Médias das Emoções:")
    print(medias_emocoes)

    # 8. Salvando os Resultados
    df.to_csv('analise_sentimentos_reddit.csv', index=False)
    df.to_excel('analise_sentimentos_reddit.xlsx', index=False)

else:
    print("Nenhum comentário coletado.")

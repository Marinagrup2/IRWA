import os
from json import JSONEncoder
import random

# pip install httpagentparser
import httpagentparser  # for getting the user agent as json
import nltk
from flask import Flask, render_template, session
from flask import request

from myapp.analytics.analytics_data import AnalyticsData, ClickedDoc
from myapp.search.load_corpus import load_corpus
from myapp.search.objects import Document, StatsDocument
from myapp.search.search_engine import SearchEngine

from datetime import datetime
import altair as alt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import folium
from collections import Counter
from nltk.corpus import stopwords
import nltk
from myapp.analytics.analytics_database import AnalyticsDatabase


# Crear instancia de la base de datos
analytics_db = AnalyticsDatabase()

# Inicializar la base de datos
analytics_db.init_db()

##############3
# En caso de que por cada sesion queramos resetear el numero de clicks en count_clicks.py descomentar el codigo siguiente
'''
def reset_clicks():
    """
    Reset the clicks table by deleting all records.
    """
    query = "DELETE FROM click"
    with sqlite3.connect(analytics_db.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()
    print("Clicks table has been reset.")

# Llama a esta función al iniciar la aplicación
reset_clicks()
'''
# Ensure you have the necessary NLTK resources
nltk.download('stopwords')

# *** for using method to_json in objects ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)


_default.default = JSONEncoder().default
JSONEncoder.default = _default

# end lines ***for using method to_json in objects ***

# instantiate the Flask application
app = Flask(__name__)

# random 'secret_key' is used for persisting data in secure cookie
app.secret_key = 'afgsreg86sr897b6st8b76va8er76fcs6g8d7'
# open browser dev tool to see the cookies
app.session_cookie_name = 'IRWA_SEARCH_ENGINE'

# instantiate our search engine
search_engine = SearchEngine()

# instantiate our in memory persistence
analytics_data = AnalyticsData()

# print("current dir", os.getcwd() + "\n")
# print("__file__", __file__ + "\n")
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
# print(path + ' --> ' + filename + "\n")
# load documents corpus into memory.
file_path = path + "\\farmers-protest-tweets.json"

# file_path = "../../tweets-data-who.json"
corpus = load_corpus(file_path)
print("loaded corpus. first elem:", list(corpus.values())[0])


# Home URL "/"
@app.route('/')
def index():
    print("starting home url /...")

    # flask server creates a session by persisting a cookie in the user's browser.
    # the 'session' object keeps data between multiple requests
    session['some_var'] = "IRWA 2021 home"

    user_agent = request.headers.get('User-Agent')
    print("Raw user browser:", user_agent)

    user_ip = request.remote_addr
    agent = httpagentparser.detect(user_agent)

    print("Remote IP: {} - JSON user browser {}".format(user_ip, agent))

    print(session)

    return render_template('index.html', page_title="Welcome")

'''
@app.route('/search', methods=['POST'])
def search_form_post():
    search_query = request.form['search-query']

    session['last_search_query'] = search_query

    search_id = analytics_data.save_query_terms(search_query)

    results = search_engine.search(search_query, search_id, corpus)

    found_count = len(results)
    session['last_found_count'] = found_count
    
    print(session)

    return render_template('results.html', results_list=results, page_title="Results", found_counter=found_count)
'''
@app.route('/search', methods=['POST'])
def search_form_post():
    search_query = request.form['search-query']
    session_id = session['session_id']

    # Guardar la consulta en memoria
    search_id = analytics_data.save_query_terms(search_query)

    # Guardar la consulta en SQLite
    analytics_db.save_query(
        session_id=session_id,
        query_text=search_query,
        timestamp=datetime.now().isoformat()
    )

    # Realizar la búsqueda
    results = search_engine.search(search_query, search_id, corpus)

    return render_template('results.html', results_list=results, page_title="Results", found_counter=len(results))


'''
@app.before_request
def track_session_start():
    """Track session start."""
    if 'session_id' not in session:
        session_id = random.randint(0, 100000)
        session['session_id'] = session_id
        analytics_db.save_session(
            session_id=session_id,
            ip=request.remote_addr,
            user_agent=request.headers.get('User-Agent'),
            start_time=datetime.now().isoformat()
        )

'''
@app.before_request
def track_session_start():
    """Track session start."""
    if 'session_id' not in session:
        session_id = random.randint(0, 100000)
        session['session_id'] = session_id
        
        # Guardar la sesión en memoria
        analytics_data.save_session(
            session_id=session_id,
            ip=request.remote_addr,
            user_agent=request.headers.get('User-Agent'),
            start_time=datetime.now().isoformat()
        )

        # Guardar la sesión en SQLite
        analytics_db.save_session(
            session_id=session_id,
            ip=request.remote_addr,
            user_agent=request.headers.get('User-Agent'),
            start_time=datetime.now().isoformat()
        )


@app.teardown_request
def track_session_end(exception=None):
    """Track session end time."""
    if 'session_id' in session:
        end_time = datetime.now().isoformat()

        # Finalizar la sesión en memoria
        analytics_data.end_session(session['session_id'], end_time)

        # Finalizar la sesión en SQLite
        analytics_db.end_session(session['session_id'], end_time)
    #session['session_end'] = datetime.now().isoformat()



'''
@app.route('/doc_details', methods=['GET'])
def doc_details():
    # getting request parameters:
    # user = request.args.get('user')

    print("doc details session: ")
    print(session)
    

    res = session["some_var"]

    print("recovered var from session:", res)

    # get the query string parameters from request
    clicked_doc_id = request.args["id"]
    search_id = int(request.args["search_id"])  # Transform to Integer
    timestamp = datetime.now() #time when user clicked on that doc

    p1 = int(request.args["search_id"])  # transform to Integer
    p2 = int(request.args["param2"])  # transform to Integer
    print("click in id={}".format(clicked_doc_id))

    # Update analytics_data with clicked document and timestamp
    analytics_db.save_click(
        session_id=session_id,
        doc_id=clicked_doc_id,
        timestamp=datetime.now().isoformat()
    )
    print("fact_clicks count for id={} is {}".format(clicked_doc_id, analytics_data.fact_clicks[clicked_doc_id]))

    return render_template('doc_details.html')
'''

@app.route('/doc_details', methods=['GET'])
def doc_details():
    """
    Handle document click event and save it to the database.
    """
    print("doc details session: ")
    print(session)

    # Recuperar la variable 'session_id' desde la sesión de Flask
    session_id = session.get('session_id')  # Obtén session_id de la sesión
    if not session_id:
        return "Session ID not found", 400

    # Obtén los parámetros de la solicitud
    clicked_doc_id = request.args.get("id")
    search_id = request.args.get("search_id")  # Aquí no necesitas convertirlo si ya es un string
    timestamp = datetime.now()  # Hora en que el usuario hizo clic en el documento

    if not clicked_doc_id or not search_id:
        return "Missing parameters", 400

    # Guarda el clic en la base de datos
    analytics_db.save_click(
        session_id=session_id,
        doc_id=clicked_doc_id,
        timestamp=timestamp.isoformat()
    )
    print(f"Click registrado: {clicked_doc_id} en la sesión {session_id}")

    return f"Document {clicked_doc_id} clicked and registered.", 200


@app.route('/stats', methods=['GET'])
def stats():
    """
    Show simple statistics example. ### Replace with dashboard ###
    :return:
    """

    docs = []
    # ### Start replace with your code ###

    for doc_id in analytics_data.fact_clicks:
        row: Document = corpus[int(doc_id)]
        count = analytics_data.fact_clicks[doc_id]
        doc = StatsDocument(row.id, row.title, row.description, row.doc_date, row.url, count)
        docs.append(doc)

    # simulate sort by ranking
    docs.sort(key=lambda doc: doc.count, reverse=True)
    return render_template('stats.html', clicks_data=docs)
    # ### End replace with your code ###

'''
@app.route('/stats', methods=['GET'])
def stats():
    session_start = session.get('session_start')
    session_end = session.get('session_end')
    total_time = "N/A"

    #session time
    if session_start and session_end:
        start_time = datetime.fromisoformat(session_start)
        end_time = datetime.fromisoformat(session_end)
        total_time = str(end_time - start_time)  # Total session time

    #clicked documents 
    clicked_docs = []
    for doc_id, timestamp in analytics_data.fact_clicks.items(): #s'haura de canviar pels clicked documents segons sessió
        row: Document = corpus[int(doc_id)]
        clicked_docs.append(StatsDocument(row.id, row.title, row.description, row.doc_date, row.url, timestamp))

    # queries
    queries = [{"query": query} for query in analytics_data.fact_queries] #aqui tmb canviar per queries nomes de la sessió

    df = pd.DataFrame(clicked_docs)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds()

    # Create the Altair chart
    chart = alt.Chart(df).mark_line(point=True).encode(
        x='timestamp:T',
        y='time_diff:Q',
        tooltip=['doc_id', 'title', 'time_diff']
    ).properties(
        title='Time Difference Between Document Clicks',
        width=600,
        height=400
    )
    chart_html = chart.to_html()

    return render_template('stats.html',
                           clicks_data=clicked_docs,
                           queries=queries,
                           total_time=total_time,
                           chart_html=chart_html)
'''

@app.route('/dashboard', methods=['GET'])
def dashboard():
    visited_docs = []
    print(analytics_data.fact_clicks.keys())
    for doc_id in analytics_data.fact_clicks.keys():
        d: Document = corpus[int(doc_id)]
        doc = ClickedDoc(doc_id, d.description, analytics_data.fact_clicks[doc_id])
        visited_docs.append(doc)

    # simulate sort by ranking
    visited_docs.sort(key=lambda doc: doc.counter, reverse=True)

    for doc in visited_docs: print(doc)
    return render_template('dashboard.html', visited_docs=visited_docs)

'''
@app.route('/dashboard', methods=['GET'])
def dashboard():
    # Prepare the clicked documents data
    visited_docs = []
    for doc_id in analytics_data.fact_clicks.keys():
        d: Document = corpus[int(doc_id)]
        doc = ClickedDoc(doc_id, d.description, analytics_data.fact_clicks[doc_id])
        visited_docs.append(doc)

    # Simulate sorting by ranking (most clicked first)
    visited_docs.sort(key=lambda doc: doc.counter, reverse=True)

    # Prepare query data for analysis (number of terms, most common words)
    query_terms = [query.split() for query in analytics_data.fact_queries]  # List of lists of words
    all_query_terms = [term for sublist in query_terms for term in sublist]  # Flatten list

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_query_terms = [term for term in all_query_terms if term.lower() not in stop_words]

    # Word cloud generation
    wordcloud = WordCloud(width=800, height=400).generate(' '.join(filtered_query_terms))
    wordcloud_img = wordcloud.to_image()

    # Histogram of number of terms in queries
    query_lengths = [len(query.split()) for query in analytics_data.fact_queries]
    query_length_histogram = plt.figure(figsize=(6, 4))
    sns.histplot(query_lengths, bins=range(min(query_lengths), max(query_lengths) + 1), kde=False)
    plt.title('Histogram of Query Lengths')
    plt.xlabel('Number of Terms in Query')
    plt.ylabel('Frequency')
    query_length_histogram_img = 'static/images/histogram.png'
    query_length_histogram.savefig(query_length_histogram_img)

    # Browser and OS distribution (you'll need to parse this from the user-agent data stored earlier)
    browsers = [agent.get('browser', 'unknown') for agent in analytics_data.fact_clicks.values()]
    operating_systems = [agent.get('os', 'unknown') for agent in analytics_data.fact_clicks.values()]

    # Plot browser distribution
    browser_count = Counter(browsers)
    browser_count_plot = plt.figure(figsize=(8, 6))
    sns.barplot(x=list(browser_count.keys()), y=list(browser_count.values()))
    plt.title('Browser Distribution')
    plt.xlabel('Browser')
    plt.ylabel('Count')
    browser_count_plot_img = 'static/images/browser_distribution.png'
    browser_count_plot.savefig(browser_count_plot_img)

    # Plot OS distribution
    os_count = Counter(operating_systems)
    os_count_plot = plt.figure(figsize=(8, 6))
    sns.barplot(x=list(os_count.keys()), y=list(os_count.values()))
    plt.title('Operating System Distribution')
    plt.xlabel('OS')
    plt.ylabel('Count')
    os_count_plot_img = 'static/images/os_distribution.png'
    os_count_plot.savefig(os_count_plot_img)

    # Plot visitors per day (using a timestamp from clicks)
    click_times = [doc.timestamp for doc in visited_docs]
    visitor_dates = pd.to_datetime(click_times).dt.date
    visitors_per_day = pd.Series(visitor_dates).value_counts().sort_index()
    visitors_per_day_plot = visitors_per_day.plot(kind='line', figsize=(10, 6))
    visitors_per_day_plot.set_title('Visitors Per Day')
    visitors_per_day_plot.set_xlabel('Date')
    visitors_per_day_plot.set_ylabel('Number of Visitors')
    visitors_per_day_plot_img = 'static/images/visitors_per_day.png'
    visitors_per_day_plot.get_figure().savefig(visitors_per_day_plot_img)

    # Collect IP-related data (IP, Country, City, Browser, OS)
    visitor_data = []
    for ip, data in analytics_data.fact_clicks.items():
        country = data['country']
        city = data['city']
        browser = data['browser']
        os = data['os']
        
        # Create a row for each visitor
        visitor_data.append({
            'ip': ip,
            'country': country,
            'city': city,
            'browser': browser,
            'os': os
        })
    # Render the dashboard template with all the data and visualizations
    return render_template('dashboard.html', 
                           visited_docs=visited_docs,
                           visitor_data=visitor_data,
                           wordcloud_img=wordcloud_img,
                           query_length_histogram_img=query_length_histogram_img,
                           browser_count_plot_img=browser_count_plot_img,
                           os_count_plot_img=os_count_plot_img,
                           visitors_per_day_plot_img=visitors_per_day_plot_img)
'''

@app.route('/sentiment')
def sentiment_form():
    return render_template('sentiment.html')


@app.route('/sentiment', methods=['POST'])
def sentiment_form_post():
    text = request.form['text']
    nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    score = ((sid.polarity_scores(str(text)))['compound'])
    return render_template('sentiment.html', score=score)


if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=True)

'''
@app.route('/view_clicks')
def view_clicks():
    """
    Display the list of clicks from the database.
    """
    query = "SELECT * FROM click"
    with sqlite3.connect(analytics_db.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        clicks = cursor.fetchall()
    return render_template('view_clicks.html', clicks=clicks)
'''
@app.route('/clicks_count')
def clicks_count():
    """
    Display the total number of clicks.
    """
    query = "SELECT COUNT(*) FROM click"
    with sqlite3.connect(analytics_db.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        total_clicks = cursor.fetchone()[0]
    return f"Total clicks: {total_clicks}"

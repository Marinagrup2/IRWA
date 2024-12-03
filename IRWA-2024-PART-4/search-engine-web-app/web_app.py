import os
from json import JSONEncoder
import random
import pickle

# pip install httpagentparser
import httpagentparser  # for getting the user agent as json
import nltk
from flask import Flask, render_template, session
from flask import request

from myapp.analytics.analytics_data import AnalyticsData, ClickedDoc
from myapp.search.objects import ResultItem, Document
from myapp.search.load_corpus import load_corpus
from myapp.search.objects import Document, StatsDocument
from myapp.search.search_engine import SearchEngine
from myapp.search.algorithms import create_index_tf_idf

from datetime import datetime, timedelta
import altair as alt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import folium
from collections import Counter
from nltk.corpus import stopwords
import nltk
from myapp.analytics.analytics_database import AnalyticsCSV
import sqlite3
import requests
import csv
import geoip2.database  
import plotly.express as px


# Initialize the analytics CSV handler
analytics_csv = AnalyticsCSV('analytics.csv')

# Initialize the CSV file (creates the file with headers if it doesn't exist)
analytics_csv.init_csv()

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
file_path = path + "/tweets-data-who.json"
CSV_FILE_PATH = path +  "/analytics.csv"
# Path to the GeoLite2 database
GEOIP_DB_PATH = path + '/GeoLite2-City.mmdb'

corpus = load_corpus(file_path)
print("loaded corpus. first elem:", list(corpus.values())[0])

doc_for_idx = path + '/index.pkl'
doc_for_tf = path + '/tf.pkl'
doc_for_idf = path + '/idf.pkl'

if os.path.exists(doc_for_idx) and os.path.exists(doc_for_tf) and os.path.exists(doc_for_idf):
    with open(doc_for_idx, 'rb') as archivo:
        idx = pickle.load(archivo)
    with open(doc_for_tf, 'rb') as archivo:
        tf = pickle.load(archivo)
    with open(doc_for_idf, 'rb') as archivo:
        idf = pickle.load(archivo)
else:
    idx, tf, idf = create_index_tf_idf(corpus)
    with open(doc_for_idx, 'wb') as archivo:
        pickle.dump(idx, archivo)
    with open(doc_for_tf, 'wb') as archivo:
        pickle.dump(tf, archivo)
    with open(doc_for_idf, 'wb') as archivo:
        pickle.dump(idf, archivo)

def get_last_timestamp_for_session(file_path, session_id):
    """
    Reads the CSV file and retrieves the last timestamp for a given session_id.
    """
    last_timestamp = None
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)  # Use DictReader for easier column handling
        for row in reader:
            if row['session_id'] == str(session_id):
                # Update the last timestamp if found
                last_timestamp = row['timestamp'] #reads all rows for the session id and stores the last timestamp
    return last_timestamp

def generate_realistic_ip():
    """
    Generate a realistic, geographically valid IPv4 address.
    """
    # Define common public IP address ranges
    public_ip_ranges = [
        (1, 126),       # APNIC region (Asia-Pacific)
        (128, 191),     # RIPE region (Europe/Middle East/Central Asia)
        (192, 223),     # ARIN/LACNIC region (Americas)
    ]
    
    # Pick a random range
    first_octet_range = random.choice(public_ip_ranges)
    first_octet = random.randint(first_octet_range[0], first_octet_range[1])
    
    # Generate other octets
    second_octet = random.randint(0, 255)
    third_octet = random.randint(0, 255)
    fourth_octet = random.randint(0, 255)
    
    return f"{first_octet}.{second_octet}.{third_octet}.{fourth_octet}"

# Home URL "/"
@app.route('/')
def index():
    print("starting home url /...")

    # flask server creates a session by persisting a cookie in the user's browser.
    # the 'session' object keeps data between multiple requests
    session['some_var'] = "IRWA 2021 home"

    user_agent = request.headers.get('User-Agent')
    print("Raw user browser:", user_agent)

    ip_address = generate_realistic_ip()
    agent = httpagentparser.detect(user_agent)

    print("Remote IP: {} - JSON user browser {}".format(ip_address, agent))
    print("------------------")
    print(agent)
    # Extracting names
    browser = agent['browser']['name']
    operating_system = agent['os']['name']

    print("Browser:", browser)
    print("Operating System:", operating_system)

    # Check if session_id is in session and if it's still valid
    current_time = datetime.now()
    session_id = session.get('session_id')

    if session_id:
        # Get the last timestamp for the session from the CSV
        last_timestamp_str = get_last_timestamp_for_session(analytics_csv.file_path, session_id)
        if last_timestamp_str:
            # Convert the timestamp to a datetime object
            last_timestamp = datetime.fromisoformat(last_timestamp_str.split(",")[0])  # Remove milliseconds
            if current_time - last_timestamp <= timedelta(hours=2):
                # Session is still valid; no need to create a new one
                return render_template('index.html', page_title="Welcome")
    
    # If no session_id exists or it has expired, create a new session
    session['session_id'] = random.randint(1, 100000)
    start_time = current_time.isoformat()

    analytics_csv.save_session(
        session_id=session['session_id'],
        ip=ip_address,
        user_agent=user_agent,
        start_time=start_time
    )

    return render_template('index.html', page_title="Welcome")


@app.route('/search', methods=['POST'])
def search_form_post():
    search_query = request.form['search-query']
    session_id = session['session_id']
    timestamp=datetime.now().isoformat()

    # Guardar la consulta en memoria
    search_id = analytics_data.save_query_terms(search_query)

    # Guardar la consulta en SQLite
    analytics_csv.save_query(
        session_id=session_id,
        query_text=search_query,
        timestamp=timestamp
    )

    # Realizar la búsqueda
    search_method = request.form['search_method']
    results = search_engine.search(search_query, search_id, corpus, idx, tf, idf, search_method)

    # Only show the top 20 result (so computer doesn't crash)
    ranked_docs = []
    top = 20
    
    for d_id in results[:top]:
        item: Document = corpus[d_id]
        ranked_docs.append(ResultItem(item.id, item.title, item.description, item.doc_date, item.url, "doc_details?id={}&search_id={}&param2=2".format(item.id, search_id)))

    return render_template('results.html', results_list=ranked_docs, page_title="Results", found_counter=len(results))

@app.before_request
def track_session_start():
    if 'session_id' not in session:
        session['session_id'] = random.randint(1, 100000)  # Generar un session_id único
        user_agent = request.headers.get('User-Agent')
        ip_address = request.remote_addr
        start_time = datetime.now().isoformat()

        # Guarda los detalles en la tabla analytics
        analytics_csv.save_session(
            session_id=session['session_id'],
            ip=ip_address,
            user_agent=user_agent,
            start_time=start_time
        )

@app.route('/doc_details', methods=['GET'])
def doc_details():
    session_id = session.get('session_id')
    doc_id = request.args.get('id') # ID del documento como entero
    search_id = request.args.get('search_id')  # Este valor parece no usarse directamente
    timestamp = datetime.now().isoformat()

    # Validar que el doc_id exista en el corpus
    if doc_id not in corpus:
        return "Documento no encontrado", 404

    # Extraer dinámicamente el título y la descripción del documento
    document = corpus[doc_id]
    title = document.title
    description = document.description

    # Guarda el clic en la tabla analytics
    analytics_csv.save_click(
        session_id=session_id,
        doc_id=doc_id,
        title=title,
        description=description,
        timestamp=timestamp
    )

    tweet: Document = corpus[doc_id]

    return render_template('doc_details.html', title=title, description=description, tweet=tweet)


@app.route('/stats', methods=['GET'])
def stats():
    session_id = session.get('session_id')  # Get the current session ID
    
    if not session_id:
        return "Session ID not found.", 400

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(CSV_FILE_PATH)

    # Filter data for the current session
    session_data = df[df['session_id'] == session_id]

    # Calculate total session time (from first and last clicked documents)
    total_time = "N/A"
    clicked_documents = session_data[session_data['doc_id'].notna()]
    if not clicked_documents.empty:
        timestamps = pd.to_datetime(clicked_documents['timestamp'])
        # Find the earliest timestamp (first event of the session)
        first_timestamp = timestamps.min()
        current_time = pd.Timestamp.now()  # Current time
        time_difference = current_time - first_timestamp  # Timedelta object

        # Convert the timedelta to hours, minutes, and seconds
        total_seconds = int(time_difference.total_seconds())  # Total seconds as integer
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        # Format as "xx hours xx minutes xx seconds"
        total_time = f"{hours} hours {minutes} minutes {seconds} seconds"


    # Fetch clicked document details for visualization
    clicked_documents = clicked_documents[['doc_id', 'title', 'description', 'timestamp']].sort_values(by='timestamp')

    if not clicked_documents.empty:
        # Ensure document IDs or titles are sorted in the order of clicks
        clicked_documents['doc_label'] = clicked_documents['doc_id'].astype(str)

        clicked_documents['timestamp'] = pd.to_datetime(clicked_documents['timestamp'])

        # Calculate time differences in minutes
        clicked_documents['time_diff_minutes'] = clicked_documents['timestamp'].diff().dt.total_seconds() / 60

        # Replace NaN in time_diff_minutes with 0 for the first document
        clicked_documents['time_diff_minutes'] = clicked_documents['time_diff_minutes'].fillna(0)

        # Convert timestamps to a readable format for tooltips
        clicked_documents['timestamp_readable'] = clicked_documents['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Create the Altair chart with documents on X-axis and time differences on Y-axis
        chart = alt.Chart(clicked_documents).mark_line(point=True).encode(
            x=alt.X('doc_label:N', title='Clicked Documents', sort=None),  # Categorical X-axis
            y=alt.Y('time_diff_minutes:Q', title='Time Difference Between Clicks (Minutes)'),  # Quantitative Y-axis
            tooltip=['doc_label', 'timestamp_readable', 'time_diff_minutes']  # Add tooltips for more context
        ).properties(
            title='Time Differences Between Clicked Documents',
            width=600,
            height=400
        )

        chart_html = chart.to_html()

    else:
        chart_html = "<p>No data available for clicked documents.</p>"

    # Fetch queries for the current session from the CSV file
    queries = session_data[session_data['query'].notna()]

    queries_list = [{"query": query} for query in queries['query'].tolist()]

    # Prepare the table content for the clicked documents
    table_data = []
    for index, row in clicked_documents.iterrows():
        table_data.append({
            'doc_id': row['doc_id'],
            'title': row['title'],
            'description': row['description'],
            'timestamp': row['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'time_diff_minutes': row['time_diff_minutes']
        })

    # Prepare the data for rendering
    return render_template(
        'stats.html',
        clicks_data=table_data,
        queries=queries_list,
        total_time=total_time,
        chart_html=chart_html
    )

# Function to get country info from IP
def get_country_from_ip(ip):
    try:
        # Request the geolocation data using ipinfo.io
        response = requests.get(f'http://ipinfo.io/{ip}/json')
        data = response.json()
        country = data.get('country', 'Unknown')
        city = data.get('city', 'Unknown')
        return country, city
    except requests.RequestException:
        # Return 'Unknown' if the request fails
        return 'Unknown', 'Unknown'
    
@app.route('/dashboard', methods=['GET'])
def dashboard():
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(CSV_FILE_PATH)

    # Extract IP addresses and resolve them to countries
    # Extract IP addresses and resolve them to countries
    geo_data = []
    with geoip2.database.Reader(GEOIP_DB_PATH) as reader:
        for ip in df['ip_address'].dropna().unique():
            try:
                response = reader.city(ip)
                country = response.country.name
                geo_data.append((ip, country))
            except geoip2.errors.AddressNotFoundError:
                geo_data.append((ip, 'Unknown'))

    # Convert geo_data to DataFrame and merge with the original session data
    geo_df = pd.DataFrame(geo_data, columns=['ip_address', 'country'])
    df = df.merge(geo_df, on='ip_address', how='left')

    # Count sessions by country
    country_session_count = df.groupby('country')['session_id'].nunique().reset_index()
    country_session_count.columns = ['country', 'sessions']

    # Create a choropleth map using Plotly
    fig = px.choropleth(
        country_session_count,
        locations='country',
        locationmode='country names',
        color='sessions',
        title='Sessions by Country',
        color_continuous_scale='Viridis',
        labels={'sessions': 'Number of Sessions'}
    )

    # Save the map as an HTML file
    map_html_path = 'static/maps/sessions_by_country.html'
    fig.write_html(map_html_path)

    # Prepare the IP-to-Country table to display
    ip_country_table = geo_df.to_html(classes='table table-striped', index=False)


    # Filter out documents with no doc_id (for clicked docs analysis)
    clicked_docs = df[df['doc_id'].notna()]

    # Calculate the count of clicks for each document (doc_id, title, description)
    clicked_docs_count = clicked_docs.groupby(['doc_id', 'title', 'description']).size().reset_index(name='counter')
    clicked_docs_count = clicked_docs_count.sort_values(by='counter', ascending=False)

    # Prepare the visited documents list with doc_id, description, and click count
    visited_docs = []
    for _, row in clicked_docs_count.iterrows():
        doc = ClickedDoc(row['doc_id'], row['description'], row['counter'])
        visited_docs.append(doc)

    # Prepare query data for analysis (number of terms, most common words)
    query_terms = []
    queries = df[df['query'].notna()]['query']

    # Split each query into words
    query_terms = [query.split() for query in queries.tolist()]  # List of lists of words

    # Flatten the list of query terms and remove stopwords
    all_query_terms = [term for sublist in query_terms for term in sublist]
    stop_words = set(stopwords.words('english'))
    filtered_query_terms = [term for term in all_query_terms if term.lower() not in stop_words]

    # Count the frequency of each term
    word_counts = Counter(filtered_query_terms)

    # Generate word cloud using word frequencies
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)

    # Save the word cloud to an image file
    wordcloud_img_path = 'static/images/wordcloud.png'
    wordcloud.to_file(wordcloud_img_path)

    # Query length analysis
    query_lengths = [len(query.split()) for query in queries.tolist()]

    # Count the frequency of each query length
    query_length_counts = Counter(query_lengths)

    # Prepare data for plotting
    length_data = list(query_length_counts.items())
    length_data.sort()  # Sort by query length
    
    # Separate lengths and counts for plotting
    lengths, counts = zip(*length_data)

    # Create a plot to show the frequency of each query length
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(lengths), y=list(counts), color='red')
    plt.title('Distribution of Query Lengths')
    plt.xlabel('Query Length (Number of Terms)')
    plt.ylabel('Frequency (Count of Queries)')
    
    # Save the plot
    query_length_histogram_img = 'static/images/histogram.png'
    plt.savefig(query_length_histogram_img)
    plt.close()

    # Get the most common browser and OS per session
    browsers = df[df['browser'].notna()]['browser'].tolist()
    operating_systems = df[df['operating_system'].notna()]['operating_system'].tolist()

    # Count the occurrences of each browser and operating system
    browser_count = Counter(browsers)
    os_count = Counter(operating_systems)

    # Plot browser distribution
    browser_count_plot = plt.figure(figsize=(8, 6))
    sns.barplot(x=list(browser_count.keys()), y=list(browser_count.values()))
    plt.title('Browser Distribution')
    plt.xlabel('Browser')
    plt.ylabel('Count')
    browser_count_plot_img = 'static/images/browser_distribution.png'
    browser_count_plot.savefig(browser_count_plot_img)
    plt.close()

    # Plot OS distribution
    os_count_plot = plt.figure(figsize=(8, 6))
    sns.barplot(x=list(os_count.keys()), y=list(os_count.values()))
    plt.title('Operating System Distribution')
    plt.xlabel('OS')
    plt.ylabel('Count')
    os_count_plot_img = 'static/images/os_distribution.png'
    os_count_plot.savefig(os_count_plot_img)
    plt.close()

    # Query session IDs and timestamps from the CSV file
    session_data = df[df['timestamp'].notna()][['session_id', 'timestamp']]

    # Ensure timestamps are in datetime format and extract dates
    session_data['timestamp'] = pd.to_datetime(session_data['timestamp'])
    session_data['date'] = session_data['timestamp'].dt.date

    # Count unique session IDs per day
    unique_sessions_per_day = session_data.groupby('date')['session_id'].nunique()

    # Plot the data
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=unique_sessions_per_day.index, y=unique_sessions_per_day.values, marker='o', label='Unique Sessions')

    # Set title and labels
    plt.title('Unique Sessions Per Day')
    plt.xlabel('Date')
    plt.ylabel('Number of Unique Sessions')

    # Set the x-axis ticks to only include the dates in the dataset
    plt.xticks(ticks=unique_sessions_per_day.index, labels=unique_sessions_per_day.index, rotation=45)
    plt.grid(True)

    # Save the plot
    unique_sessions_plot_img = 'static/images/unique_sessions_per_day.png'
    plt.savefig(unique_sessions_plot_img)
    plt.close()

    # Render the dashboard template with all the data and visualizations
    return render_template('dashboard.html', 
                           visited_docs=visited_docs,
                           wordcloud_img_path=wordcloud_img_path,
                           query_length_histogram_img=query_length_histogram_img,
                           browser_count_plot_img=browser_count_plot_img,
                           os_count_plot_img=os_count_plot_img,
                           unique_sessions_plot_img=unique_sessions_plot_img,
                           sessions_by_country_map=map_html_path,
                           ip_country_table=ip_country_table)


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

# ================================
# AI NEWS AGENT - PRODUCTION VERSION
# ================================

import os
import feedparser
from newspaper import Article
from transformers import pipeline
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================================
# LOAD MODELS
# ================================

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# NOTE: keeping your summarizer unused since you're doing headline-only briefing
# (you can remove it entirely safely)

print("Models Loaded")

# ================================
# RSS FEEDS
# ================================

rss_feeds = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://rss.cnn.com/rss/edition.rss",
    "https://feeds.reuters.com/reuters/topNews",
    "https://www.thehindu.com/news/national/feeder/default.rss",
    "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms"
]

# ================================
# COLLECT LINKS
# ================================

article_links = []

for feed_url in rss_feeds:
    feed = feedparser.parse(feed_url)

    for entry in feed.entries[:10]:
        if entry.link not in article_links:
            article_links.append(entry.link)

print("Articles fetched:", len(article_links))

# ================================
# SCRAPE ARTICLES
# ================================

news_articles = []

ignore_words = ["advertisement", "sponsored", "watch live", "video", "photos"]

for link in article_links:

    try:
        article = Article(link)
        article.download()
        article.parse()

        title = article.title or ""
        text = article.text or ""

        if len(text) < 500:
            continue

        if any(word in title.lower() for word in ignore_words):
            continue

        news_articles.append({
            "title": title.strip(),
            "link": link
        })

    except:
        continue

print("Clean articles:", len(news_articles))

# ================================
# DEDUPLICATION
# ================================

if len(news_articles) > 1:

    titles = [a["title"] for a in news_articles]

    vectorizer = TfidfVectorizer().fit_transform(titles)
    similarity = cosine_similarity(vectorizer)

    used = set()
    unique_articles = []

    for i in range(len(news_articles)):

        if i in used:
            continue

        unique_articles.append(news_articles[i])

        for j in range(i + 1, len(news_articles)):
            if similarity[i][j] > 0.6:
                used.add(j)

    news_articles = unique_articles

print("After dedupe:", len(news_articles))

# ================================
# CATEGORY BUCKETS
# ================================

categorized = {
    "🌍 GLOBAL NEWS": [],
    "💼 BUSINESS & ECONOMY": [],
    "🤖 TECHNOLOGY & AI": [],
    "🇮🇳 INDIA": [],
    "📌 OTHER": []
}

global_keywords = [
    "war", "ukraine", "russia", "china", "israel",
    "gaza", "nato", "iran", "un", "world", "global",
    "military", "attack", "conflict"
]

# ================================
# CLASSIFICATION
# ================================

labels = [
    "Global News",
    "Business & Economy",
    "Technology & AI",
    "India"
]

for article in news_articles:

    title = article["title"]
    low = title.lower()

    assigned = False

    # FORCE GLOBAL
    if any(k in low for k in global_keywords):
        categorized["🌍 GLOBAL NEWS"].append(article)
        continue

    try:
        result = classifier(title, labels)
        category = result["labels"][0]

    except:
        category = "Other"

    if category == "Business & Economy":
        categorized["💼 BUSINESS & ECONOMY"].append(article)

    elif category == "Technology & AI":
        categorized["🤖 TECHNOLOGY & AI"].append(article)

    elif category == "India":
        categorized["🇮🇳 INDIA"].append(article)

    elif category == "Global News":
        categorized["🌍 GLOBAL NEWS"].append(article)

    else:
        categorized["📌 OTHER"].append(article)

# ================================
# BUILD EMAIL CONTENT
# ================================

daily_summary = ""

order = [
    "🌍 GLOBAL NEWS",
    "💼 BUSINESS & ECONOMY",
    "🤖 TECHNOLOGY & AI",
    "🇮🇳 INDIA",
    "📌 OTHER"
]

for cat in order:

    if len(categorized[cat]) == 0:
        continue

    daily_summary += f"\n{cat}\n\n"

    for article in categorized[cat][:6]:
        daily_summary += f"• {article['title']}\n  {article['link']}\n\n"

print("Summary ready")

# ================================
# EMAIL CONFIG (ENV VARIABLES)
# ================================

sender_email = os.getenv("EMAIL_USER")
password = os.getenv("EMAIL_PASS")

receiver_email = sender_email

if not sender_email or not password:
    raise ValueError("Missing EMAIL_USER or EMAIL_PASS in environment variables")

# ================================
# CREATE EMAIL
# ================================

message = MIMEMultipart("alternative")
message["Subject"] = "📰 Daily AI News Briefing"
message["From"] = sender_email
message["To"] = receiver_email

html = f"""
<html>
<body style="font-family: Arial; padding: 20px;">

<h2>📰 Daily AI News Briefing</h2>

<pre style="white-space: pre-wrap; font-size: 14px;">
{daily_summary}
</pre>

<p style="color: gray; font-size: 12px;">
Generated automatically by AI News Agent
</p>

</body>
</html>
"""

message.attach(MIMEText(html, "html"))

# ================================
# SEND EMAIL
# ================================

try:
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender_email, password)

    server.sendmail(sender_email, receiver_email, message.as_string())
    server.quit()

    print("Email sent successfully!")

except Exception as e:
    print("Email failed:", e)

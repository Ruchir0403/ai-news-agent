# =========================================
# AI NEWS AGENT - PRODUCTION VERSION (FIXED)
# =========================================

import os
import feedparser
from newspaper import Article
from transformers import pipeline
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# DEDUPLICATION
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================
# LOAD AI MODEL
# =========================================

print("Loading AI model...")

# Using bart-large-mnli for reliable zero-shot performance
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

print("✅ AI Model Loaded")

# =========================================
# RSS FEEDS
# =========================================

rss_feeds = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://rss.cnn.com/rss/edition.rss",
    "https://feeds.reuters.com/reuters/topNews",
    "https://www.thehindu.com/news/national/feeder/default.rss",
    "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms"
]

# =========================================
# FETCH ARTICLE LINKS
# =========================================

article_links = []

for feed_url in rss_feeds:
    try:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:12]: # Slightly expanded to ensure bucket filling
            if entry.link not in article_links:
                article_links.append(entry.link)
    except Exception as e:
        print(f"RSS Error from {feed_url}: {e}")

print(f"📰 Total article links fetched: {len(article_links)}")

# =========================================
# SCRAPE ARTICLES & KEEP TEXT CONTEXT
# =========================================

news_articles = []

ignore_words = [
    "advertisement",
    "sponsored",
    "watch live",
    "video",
    "photos",
    "gallery"
]

for link in article_links:
    try:
        article = Article(link)
        article.download()
        article.parse()

        title = article.title.strip() if article.title else ""
        text = article.text.strip() if article.text else ""

        # Skip short/low-quality articles
        if len(text) < 500:
            continue

        # Ignore unwanted content
        if any(word in title.lower() for word in ignore_words):
            continue

        # FIX 1: Save a small snippet of text to give the classifier actual context
        snippet = text[:300]

        news_articles.append({
            "title": title,
            "snippet": snippet,
            "link": link
        })

    except Exception:
        pass # Silently skip scraping errors to clean terminal output

print(f"✅ Clean articles collected: {len(news_articles)}")

# =========================================
# REMOVE DUPLICATE ARTICLES
# =========================================

if len(news_articles) > 1:
    titles = [article["title"] for article in news_articles]
    vectorizer = TfidfVectorizer().fit_transform(titles)
    similarity_matrix = cosine_similarity(vectorizer)

    used_indexes = set()
    unique_articles = []

    for i in range(len(news_articles)):
        if i in used_indexes:
            continue
        unique_articles.append(news_articles[i])

        for j in range(i + 1, len(news_articles)):
            if similarity_matrix[i][j] > 0.60:
                used_indexes.add(j)

    news_articles = unique_articles

print(f"🧹 Articles after deduplication: {len(news_articles)}")

# =========================================
# EXPANDED CATEGORY BUCKETS & AI LABELS
# =========================================

categorized = {
    "🌍 GLOBAL NEWS": [],
    "💼 BUSINESS & FINANCE": [],
    "🤖 TECHNOLOGY & AI": [],
    "🇮🇳 INDIA NATIONAL": [],
    "🧬 SCIENCE & ENVIRONMENT": [],
    "⚽ SPORTS": [],
    "🎬 ENTERTAINMENT & LIFESTYLE": [],
    "📌 OTHER": []
}

# The candidate labels provided to the Zero-Shot model
labels = [
    "World News", 
    "Politics",
    "Business & Finance", 
    "Technology & AI", 
    "India National News", 
    "Science & Environment", 
    "Sports", 
    "Entertainment & Lifestyle"
]

# =========================================
# AI CLASSIFICATION
# =========================================

print("🤖 Classifying articles with semantic context...")

for article in news_articles:
    title = article["title"]
    snippet = article["snippet"]
    
    # FIX 2: Combine Title and Text Snippet so the AI understands what the story is about
    classification_text = f"Headline: {title}\nContext: {snippet}"

    try:
        result = classifier(
            classification_text,
            labels,
            multi_label=False
        )

        predicted_category = result["labels"][0]
        confidence_score = result["scores"][0]

    except Exception as e:
        predicted_category = "Other"
        confidence_score = 0

    # Low confidence fallback
    if confidence_score < 0.35: # Dialed down slightly because of more classification options
        categorized["📌 OTHER"].append(article)
        continue

    # FIX 3: Fully mapped categorical hierarchy
    if predicted_category in ["World News", "Politics"]:
        categorized["🌍 GLOBAL NEWS"].append(article)
        
    elif predicted_category == "Business & Finance":
        categorized["💼 BUSINESS & FINANCE"].append(article)
        
    elif predicted_category == "Technology & AI":
        categorized["🤖 TECHNOLOGY & AI"].append(article)
        
    elif predicted_category == "India National News":
        categorized["🇮🇳 INDIA NATIONAL"].append(article)
        
    elif predicted_category == "Science & Environment":
        categorized["🧬 SCIENCE & ENVIRONMENT"].append(article)
        
    elif predicted_category == "Sports":
        categorized["⚽ SPORTS"].append(article)
        
    elif predicted_category == "Entertainment & Lifestyle":
        categorized["🎬 ENTERTAINMENT & LIFESTYLE"].append(article)
        
    else:
        categorized["📌 OTHER"].append(article)

print("✅ Classification completed")

# =========================================
# BUILD NEWS SUMMARY
# =========================================

daily_summary = ""

category_order = [
    "🌍 GLOBAL NEWS",
    "🇮🇳 INDIA NATIONAL",
    "💼 BUSINESS & FINANCE",
    "🤖 TECHNOLOGY & AI",
    "🧬 SCIENCE & ENVIRONMENT",
    "⚽ SPORTS",
    "🎬 ENTERTAINMENT & LIFESTYLE",
    "📌 OTHER"
]

for category in category_order:
    articles = categorized[category]

    if len(articles) == 0:
        continue

    daily_summary += f"\n{category}\n\n"

    for article in articles[:6]:
        daily_summary += (
            f"• {article['title']}\n"
            f"  {article['link']}\n\n"
        )

print("✅ Summary generated")

# =========================================
# EMAIL CONFIGURATION
# =========================================

sender_email = os.getenv("EMAIL_USER")
password = os.getenv("EMAIL_PASS")
receiver_email = sender_email

if not sender_email or not password:
    raise ValueError("❌ Missing EMAIL_USER or EMAIL_PASS environment variables")

# =========================================
# CREATE EMAIL
# =========================================

today = datetime.now().strftime("%d %B %Y")
message = MIMEMultipart("alternative")
message["Subject"] = f"📰 Daily AI News Briefing | {today}"
message["From"] = sender_email
message["To"] = receiver_email

html = f"""
<html>
<body style="margin: 0; padding: 30px; background-color: #f4f6f8; font-family: 'Segoe UI', Helvetica, sans-serif;">
<div style="max-width: 900px; margin: auto; background-color: white; border-radius: 14px; padding: 35px; box-shadow: 0 4px 14px rgba(0,0,0,0.08);">
    <h1 style="margin-top: 0; color: #111827; font-size: 32px;">📰 Daily AI News Briefing</h1>
    <p style="color: #6b7280; font-size: 14px; margin-bottom: 25px;">{today}</p>
    <pre style="white-space: pre-wrap; font-size: 15px; line-height: 1.8; font-family: 'Segoe UI', Helvetica, sans-serif; color: #1f2937; background-color: #fafafa; border: 1px solid #e5e7eb; border-radius: 12px; padding: 25px;">{daily_summary}</pre>
    <p style="margin-top: 25px; font-size: 12px; color: gray;">Generated automatically by AI News Agent</p>
</div>
</body>
</html>
"""

message.attach(MIMEText(html, "html"))

# =========================================
# SEND EMAIL
# =========================================

try:
    print("📨 Sending email...")
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, message.as_string())
    server.quit()
    print("✅ Email sent successfully!")
except Exception as e:
    print("❌ Email failed")
    print(e)

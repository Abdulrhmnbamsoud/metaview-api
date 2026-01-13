# MetaView — Live News Intelligence API

MetaView is a **real-time news intelligence backend** that collects trusted news content from multiple RSS sources, stores it in a local database, and provides a **Live API** for searching, filtering, and AI-powered analysis.

This project was built as a full working system (not a mock/demo), including:
- Data ingestion (RSS scraping)
- SQLite storage
- API endpoints (FastAPI)
- AI semantic search / analytics ready pipeline
- Production deployment on cloud

---

## 🚀 Live Deployment

✅ **Live API URL:**  
https://metaview-s-693041094858.us-west1.run.app/

✅ **Swagger Docs (API UI):**  
https://metaview-693041094858.us-west1.run.app/docs

✅ **Health Check:**  
https://metaview-693041094858.us-west1.run.app/health

---

## 🎯 Key Features

### 1) Live RSS Ingestion
MetaView pulls articles from multiple trusted RSS feeds such as:
- Reuters
- BBC
- WSJ
- NYTimes
- CNN
- The Guardian
- Axios
- DW, France24, Le Monde
…and more.

The ingestion process:
- Fetches RSS feed data
- Extracts articles (headline + summary + link + publish time)
- Stores results into SQLite
- Prevents duplicates using the article URL

---

### 2) Database Storage (SQLite)
All articles are saved into a lightweight SQLite database:
- Fast local reads
- Simple deployment
- Works perfectly for MVP and production small systems

Tables:
- `articles`: stores headline, summary, source, url, published_at, etc.

---

### 3) Search & Filters API
MetaView provides live search for:
- Keyword search (headline + summary)
- Filter by source
- Filter by domain/country
- Filter by date range
- Control results count (top_k)

---

### 4) Semantic / AI Analytics Ready
The system is designed to support AI layers such as:
- Semantic search embeddings
- NLP analytics
- Sentiment analysis
- Text translation
- Category classification

This makes MetaView suitable for dashboards and monitoring systems.

---

### 5) Production Ready Deployment
MetaView is deployed and running with:
- Docker build
- Environment config
- Health monitoring endpoint

---

## 🧱 System Architecture (High-Level)

1. **RSS Sources** (News Feeds)
2. **Ingestion Engine** (fetch + parse + dedup)
3. **SQLite Database** (store all articles)
4. **FastAPI Backend** (endpoints)
5. **Frontend / Dashboard** (connects via API)

---


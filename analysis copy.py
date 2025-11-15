import json
import requests
from newspaper import Article
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions, EmotionOptions, KeywordsOptions, EntitiesOptions

# Import IBM Orchestrate SDK
#from ibm_orchestrate import OrchestrateClient

# ---------- CONFIG ----------
GOOGLE_API_KEY = "GOOGLE_API_KEY"
SEARCH_ENGINE_ID = "GOOGLE_SEARCH_ENGINE_ID"
NLU_API_KEY = "NLU_API_KEY"
NLU_API_URL = "NLU_API_URL"
ORCHESTRATE_API_KEY = "ORCHESTRATE_API_KEY"
ORCHESTRATE_URL = "ORCHESTRATE_URL"
GRANITE_MODEL_ID = "granite‑13b‑chat‑v2"


# ---------- ARTICLE EXTRACTION ----------
def extract_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        return None


# ---------- GOOGLE SEARCH ----------
def google_search(query, num_results=5):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
        "num": num_results
    }
    response = requests.get(url, params=params).json()
    return response.get("items", [])


# ---------- WATSON NLU ANALYSIS ----------
def watson_analyze(text):
    authenticator = IAMAuthenticator(NLU_API_KEY)
    nlu = NaturalLanguageUnderstandingV1(
        version="2022-04-07",
        authenticator=authenticator
    )
    nlu.set_service_url(NLU_URL)

    response = nlu.analyze(
        text=text,
        features=Features(
            sentiment=SentimentOptions(),
            emotion=EmotionOptions(),
            keywords=KeywordsOptions(limit=10),
            entities=EntitiesOptions(limit=10)
        )
    ).get_result()

    return response


def summarize_with_granite(text):
    url = ORCHESTRATE_URL  # e.g., "https://api.ibm.com/orchestrate/v1/jobs"
    headers = {
        "Authorization": f"Bearer {ORCHESTRATE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model_id": GRANITE_MODEL_ID,
        "input": f"Summarize this article:\n{text}"
    }
    resp = requests.post(url, json=payload, headers=headers)
    resp_json = resp.json()
    summary_text = resp_json.get("output_text") or resp_json.get("result") or ""
    return summary_text or "No summary available."


# ---------- COMPUTE METRICS ----------
def compute_bias(sentiment_score, emotion):
    emotion_sum = abs(emotion.get("joy", 0)) + abs(emotion.get("sadness", 0)) + abs(emotion.get("anger", 0))
    return min(1.0, (emotion_sum + abs(sentiment_score)) / 2)


def compute_fact_likelihood(keywords, entities):
    factual_indicators = len(entities)
    narrative_indicators = len(keywords)
    if factual_indicators + narrative_indicators == 0:
        return 0.5
    return factual_indicators / (factual_indicators + narrative_indicators)


# ---------- PERSPECTIVES ----------
def generate_perspectives(keywords):
    topic = keywords[0]["text"] if keywords else "the topic"
    return {
        "supporting_viewpoint": f"Supporters may feel that the article raises meaningful points about {topic} and contributes positively to understanding the issue.",
        "neutral_viewpoint": f"A neutral reader may view the article as informative and descriptive regarding {topic}.",
        "opposing_viewpoint": f"Opponents may argue the article focuses too heavily on framing or emotion when discussing {topic}, which may affect perceived objectivity."
    }


# ---------- ANALYZE ARTICLE ----------
def analyze_article(url):
    text = extract_text(url)
    if not text:
        return {"error": "Unable to extract article text."}

    nlu = watson_analyze(text)
    sentiment_score = nlu["sentiment"]["document"].get("score", 0)
    emotion = nlu["emotion"]["document"]["emotion"]
    keywords = nlu.get("keywords", [])
    entities = nlu.get("entities", [])

    # Search for related articles
    query = keywords[0]["text"] if keywords else "news article"
    search_results = google_search(query, num_results=5)

    related_analyses = []
    for r in search_results:
        analyzed = analyze_related_article(r)
        if analyzed:
            related_analyses.append(analyzed)

    supporting = [a for a in related_analyses if a["viewpoint"] == "supporting"]
    opposing = [a for a in related_analyses if a["viewpoint"] == "opposing"]
    neutral = [a for a in related_analyses if a["viewpoint"] == "neutral"]

    return {
        "bias_score": round(compute_bias(sentiment_score, emotion), 2),
        "fact_likelihood_score": round(compute_fact_likelihood(keywords, entities), 2),
        "sentiment": nlu["sentiment"]["document"],
        "emotion": emotion,
        "keywords": keywords,
        "entities": entities,
        "perspectives": generate_perspectives(keywords),
        "related_articles": {
            "supporting": supporting,
            "opposing": opposing,
            "neutral": neutral
        },
        "summary_of_viewpoints": generate_combined_viewpoint_summary(related_analyses)
    }


# ---------- ANALYZE RELATED ARTICLES ----------
def analyze_related_article(search_result):
    url = search_result.get("link")
    snippet = search_result.get("snippet", "")
    text = extract_text(url) or snippet
    if not text:
        return None

    nlu = watson_analyze(text)
    sentiment_score = nlu["sentiment"]["document"].get("score", 0)
    emotion = nlu["emotion"]["document"]["emotion"]
    keywords = nlu.get("keywords", [])
    entities = nlu.get("entities", [])

    bias = compute_bias(sentiment_score, emotion)
    fact = compute_fact_likelihood(keywords, entities)

    viewpoint = "neutral"
    if sentiment_score > 0.15:
        viewpoint = "supporting"
    elif sentiment_score < -0.15:
        viewpoint = "opposing"

    summary = summarize_with_granite(text)  # Granite summarization

    return {
        "title": search_result.get("title"),
        "url": url,
        "snippet": snippet,
        "summary": summary,
        "viewpoint": viewpoint,
        "bias_score": round(bias, 2),
        "fact_likelihood_score": round(fact, 2),
        "sentiment_score": sentiment_score
    }


# ---------- COMBINE SUMMARIES ----------
def generate_combined_viewpoint_summary(analyses):
    def summarize_group(group):
        if not group:
            return "No sources found for this viewpoint."
        texts = " ".join([a["snippet"] for a in group])
        return summarize_with_granite(texts)

    supporting = [a for a in analyses if a["viewpoint"] == "supporting"]
    opposing = [a for a in analyses if a["viewpoint"] == "opposing"]
    neutral = [a for a in analyses if a["viewpoint"] == "neutral"]

    return {
        "supporting_summary": summarize_group(supporting),
        "opposing_summary": summarize_group(opposing),
        "neutral_summary": summarize_group(neutral)
    }



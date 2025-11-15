# News-Verifier

Analyze news articles to extract text, evaluate sentiment and emotion, identify key concepts and entities, and generate summaries and viewpoints using IBM Watson NLU and a Granite model.

## Features

- Extract article text from URLs
- Analyze sentiment, emotion, keywords, and entities
- Compute bias and factual-likelihood scores
- Generate supporting, opposing, and neutral viewpoints
- Summarize articles and combined perspectives
- Search for and analyze related articles

## Usage

1. Clone the repository and install dependencies:
```bash
   pip install -r requirements.txt
```

2. Add your IBM Watson NLU credentials and Granite model API keys to the environment.

3. Run the application locally:
```bash
   python app.py
```

4. Open the web interface and enter a URL to analyze an article.

## Limitations & Next Steps

- **User Testing**: More extensive testing is needed to evaluate performance with real users
- **Readability**: Output formatting could be improved for clearer results
- **Accuracy Testing**: Further evaluation of sentiment, bias, and factual-likelihood scores is recommended
- **Results Presentation**: Consider formatting outputs (tables, summaries) for easier interpretation

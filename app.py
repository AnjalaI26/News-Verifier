from flask import Flask, render_template, request
from analysis import analyze_article

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form["url"]
        results = analyze_article(url)
        return render_template("results.html", results=results, url=url)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

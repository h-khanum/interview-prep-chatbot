from flask import Flask, render_template, request
import json

app = Flask(__name__)

with open("data/questions.json") as f:
    questions = json.load(f)

@app.route("/", methods=["GET","POST"])
def index():
    question = questions[0]
    feedback = None

    if request.method == "POST":
        user_answer = request.form["answer"]
        feedback = "Answer received"

    return render_template(
        "index.html",
        question=question["question"],
        feedback=feedback
    )

if __name__ == "__main__":
    app.run(debug=True)
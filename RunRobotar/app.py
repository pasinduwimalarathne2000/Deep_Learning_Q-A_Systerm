from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

model_name = "deepset/roberta-base-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/answer', methods=['POST'])
def answer():
    text = request.form['text']
    question = request.form['question']
    question_set = {'question': question, 'context': text}
    result = nlp(question_set)
    return jsonify({'answer': result['answer']})

@app.route('/save', methods=['POST'])
def save():
    filename = request.form['filename']
    content = f"Question: {request.form['question']}\nAnswer: {request.form['answer']}"
    with open(f"{filename}.txt", "w") as f:
        f.write(content)
    return "File saved successfully!"

if __name__ == '__main__':
    app.run(debug=True)

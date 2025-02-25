import json
import markdown
from flask import jsonify, Flask

from sentence_transformer import SentenceTransformer

app = Flask(__name__)


model_name = "sentence-transformers/all-mpnet-base-v2"
sp_model_name = "sentiment-analysis"
st = SentenceTransformer(model_name, sp_model_name)

with open("sentence_classes.json", 'r') as fp:
    sentences_dict = json.load(fp)

labeled_embeddings = {}
for label, sentences in sentences_dict.items():
    labeled_embeddings[label] = st.encode_sentences(sentences)

@app.route('/')
def home():
    with open("README.md", 'r') as f:
        text = f.read()
        html = markdown.markdown(text)
    return html

@app.route('/classify/<sentence>', methods=['GET'])
def classyfiy(sentence):
    
    label = st.classify_sentence(sentence, labeled_embeddings)
    response = {
        'sentence': sentence,
        'label': label
    }


    return jsonify(response)

@app.route('/sentiment/<sentence>', methods=['GET'])
def sentiment(sentence):
    
    sentiment = st.get_sentiment(sentence)
    response = {
        'sentence': sentence,
        'sentiment': sentiment
    }

    return jsonify(response)

if __name__ == '__main__':
    with app.test_request_context():
        print(app.url_map)
    app.run(debug=True, host='0.0.0.0', port=5000)
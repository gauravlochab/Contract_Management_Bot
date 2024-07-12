import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify
from retrieval import *
app = Flask(__name__)

@app.route('/get_response', methods=['POST'])
def api_get_response():
    data = request.get_json()
    user_input = data.get('user_input')
    chat_history = data.get('chat_history')
    chain = load_chain()
    response = get_response(user_input, chain, chat_history)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8002)

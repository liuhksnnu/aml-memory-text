from flask import Flask, request, jsonify
from flask_cors import CORS
from memory_system import MemorySystem

app = Flask(__name__)
CORS(app)  # 允许跨域，便于调试
memory_store = MemorySystem()

@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "message": "AML Memory System is running!"})

@app.route('/add', methods=['POST'])
def add_memory():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400
    user_id = data.get('user_id')
    session_id = data.get('session_id')
    content = data.get('content')
    timestamp = data.get('timestamp')
    if not all([user_id, session_id, content]):
        return jsonify({"error": "Missing required fields"}), 400
    mem_id = memory_store.add_memory(user_id, session_id, content, timestamp)
    return jsonify({"status": "success", "memory_id": mem_id})

@app.route('/search', methods=['POST'])
def search_memory():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400
    user_id = data.get('user_id')
    query = data.get('query')
    top_k = data.get('top_k', 5)
    if not user_id or not query:
        return jsonify({"error": "Missing required fields"}), 400
    results = memory_store.search_memory(user_id, query, top_k)
    return jsonify({"status": "success", "memories": results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

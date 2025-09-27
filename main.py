import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# 从 .env 文件读取 Gemini API Key
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

@app.route('/api/gemini', methods=['POST'])
def gemini_proxy():
	data = request.json
	# 这里可以添加调用 Gemini API 的逻辑
	# 示例返回
	return jsonify({
		'message': 'Gemini API Key received',
		'key': GEMINI_API_KEY,
		'data': data
	})

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000)

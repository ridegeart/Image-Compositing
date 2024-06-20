import numpy as np
import model

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return 'hello!!'

@app.route('/pothole_recognize', methods=['POST'])
def postInput():
    # 取得前端傳過來的數值
    values = request.form
    img_path = values['image_path']
    # 建立模型，預測並返回結果
    result = model.predict(img_path)
    print(result)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
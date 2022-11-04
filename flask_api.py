import base64
from cv2 import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

import backend

app = Flask(__name__)
CORS(app)
app.debug = True

SHAPE = (384, 384, 1)


def b64tocv2(b64img):
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64img), dtype=np.uint8), cv2.IMREAD_COLOR)


def cv2tob64(npimg):
    return base64.b64encode(cv2.imencode('.png', npimg)[1]).decode()


@app.route("/")
def home():
    return jsonify("running")


@app.route("/style_transfer", methods=["POST"])
def trans():
    content_img = request.json["content"]  # already base64
    style_img = request.json["style"]  # already base64
    stylized = backend.make_trans(content_img, style_img)
    output = cv2tob64(stylized)
    return jsonify({"stylized": output})


@app.route("/get_outline", methods=["POST"])
def getline():
    original_img = b64tocv2(request.json["original"])
    outlines = backend.outline(original_img)
    return jsonify({"outline": cv2tob64(outlines)})


@app.route("/gen_step", methods=["POST"])
def gen_step():
    lns = request.json["outline"]
    sort_method = request.json["sort_method"]
    steps = backend.gen_steps(cv2.cvtColor(b64tocv2(lns), cv2.COLOR_BGR2GRAY), sort_method=sort_method)
    return jsonify({"steps": steps})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=25555, ssl_context=('server.crt', 'server.key'))

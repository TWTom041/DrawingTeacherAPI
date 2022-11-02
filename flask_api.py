from flask import Flask, request, jsonify
import cv2, base64
import numpy as np
import backend
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.debug = True

SHAPE = (384, 384, 1)


def b64tocv2(b64img):
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64img), dtype=np.uint8), flag=1)


def cv2tob64(npimg):
    return base64.b64encode(cv2.imencode('.jpg', npimg)[1]).decode()


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
    print(outlines)
    return jsonify({"outline": cv2tob64(outlines)})


@app.route("/gen_step", methods=["POST"])
def gen_step():
    lns = request.values["outline"]
    sort_method = request.values["sort_method"]
    steps = backend.gen_steps(b64tocv2(lns), sort_method=sort_method)
    return jsonify({"steps": steps})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, ssl_context='adhoc')

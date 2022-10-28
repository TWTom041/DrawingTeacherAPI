from flask import Flask, request, jsonify
import cv2, base64
import numpy as np
import backend

app = Flask(__name__)

SHAPE = (384, 384, 1)





def b64tocv2(b64img):
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64img), dtype=np.uint8), flag=1)


def cv2tob64(npimg):
    return base64.b64encode(cv2.imencode('.jpg', npimg)[1]).decode()


@app.route("/")
def home():
    return "running"


@app.route("/style_transfer", methods=["POST"])
def trans():
    content_img = request.values["content"]  # already base64
    style_img = request.values["style"]  # already base64
    stylized = backend.make_trans(content_img, style_img)
    output = cv2tob64(stylized)
    return jsonify({"stylized": output})


@app.route("/get_outline", methods=["POST"])
def getline():
    original_img = b64tocv2(request.values["img"])
    outlines = backend.outline(original_img)
    return jsonify({"outline": cv2tob64(outlines)})


@app.route("/first_step", methods=["POST"])
def firstline():
    original_img = b64tocv2(request.values["content"])
    pred, prev, drew = backend.firstline(original_img)
    return jsonify({
        "pred": cv2tob64(pred),
        "prev": cv2tob64(prev),
        "drew": cv2tob64(drew),
    })


@app.route("/next_step", methods=["POST"])
def nextline():
    original_img = request.values["content"]
    cdrew = request.values["drew"]
    cprev = request.values["prev"]
    pred, prev, drew = backend.nextline(original_img, cprev, cdrew)
    return jsonify({
        "pred": cv2tob64(pred),
        "prev": cv2tob64(prev),
        "drew": cv2tob64(drew),
    })


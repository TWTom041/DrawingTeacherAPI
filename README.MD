# Drawing Teacher API
## Description
This project is the API server for Drawing Teacher App, which is [here](https://github.com/TWTom041/DrawingTeacherApp).

This project uses Flask framework as our server's backend.
## Usage
1. Clone the repo
```commandline
git clone https://github.com/TWTom041/DrawingTeacherAPI.git
```
2. Install requirements
```commandline
pip install -r requirements.txt
```
3. Set up host url and port in flask_api.py
```python
app.run(host="0.0.0.0", port=25555, ssl_context=('server.crt', 'server.key'))
```
4. Run flask_api.py
## Used things
1. https://github.com/sithu31296/semantic-segmentation
2. MobileNet-DeepLab.onnx
3. ResNet101-DUC-12-int8.onnx
4. animeGANv2
   1. https://docs.google.com/uc?export=download&id=1VPAPI84qaPUCHKHJLHiMK7BP_JE66xNe
   2. https://docs.google.com/uc?export=download&id=17XRNQgQoUAnu6SM5VgBuhqSBO4UAVNI1
   3. https://docs.google.com/uc?export=download&id=10rQfe4obW0dkNtsQuWg-szC4diBzYFXK
   4. https://docs.google.com/uc?export=download&id=1X3Glf69Ter_n2Tj6p81VpGKx7U4Dq-tI

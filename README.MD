# Drawing Teacher API
## Description
This project is the API server for Drawing Teacher App, which is [here](https://github.com/TWTom041/DrawingTeacherApp).

This project uses flask framework as our server's backend.
## Usage
1. Clone the repo
```commandline
git clone https://github.com/TWTom041/DrawingTeacherAPI.git
```
2. Install requirements
```commandline
pip install requirements.txt
```
3. Set up host url and port in flask_api.py
```python
app.run(host="0.0.0.0", port=25555, ssl_context=('server.crt', 'server.key'))
```
4. Run flask_api.py
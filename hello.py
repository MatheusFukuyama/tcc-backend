# from flask import Flask

# app = Flask(__name__)

# @app.route("/")
# def hello_world():
#     return "<p>Hello, World!</p>"

from flask import Flask

# Cria uma instância da classe Flask
app = Flask(__name__)

# Define uma rota para a página inicial
@app.route('/')
def hello_world():
    return 'Hello, World!'

# Roda o aplicativo
if __name__ == '__main__':
    app.run(debug=True)

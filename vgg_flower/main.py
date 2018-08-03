from flask import Flask, render_template





# webapp
app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def main():

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
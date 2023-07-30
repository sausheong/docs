import os
from flask import Flask, render_template, request, jsonify
from waitress import serve

from agents import Agent

# get path for static files
static_dir = os.path.join(os.path.dirname(__file__), 'static')
if not os.path.exists(static_dir):
    static_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'static')


# start server
print("\033[96mStarting qdocs at http://127.0.0.1:1338\033[0m")
qdocs = Flask(__name__, static_folder=static_dir, template_folder=static_dir)
agent = Agent()

# server landing page
@qdocs.route('/')
def landing():
    return render_template('index.html')

# run
@qdocs.route('/run', methods=['POST'])
def run():
    data = request.json
    response = agent.run(data['input'])
    return jsonify({'input': data['input'],
                    'response': response})

# reset
@qdocs.route('/reset', methods=['POST'])
def reset():
    agent.reset()
    return jsonify({
        'response': 'Agent was reset',
    })

if __name__ == '__main__':
    print("\033[93mqdocs started. Press CTRL+C to quit.\033[0m")
    # webbrowser.open("http://127.0.0.1:1338")
    serve(qdocs, port=1338, threads=16)

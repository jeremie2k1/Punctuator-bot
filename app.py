from bot_helpper.punct_msg import *
from bot_helpper.load_metrics import *
from flask import Flask, render_template, request

config, model, device = None, None, None
tokenizer, target_token2id, id2target = None, None, None
flag = False
def ready_model():
    global config, model, device
    global tokenizer, target_token2id, id2target
    global flag
    if not flag:
        metrics = metrics_loader()
        config, model, device = load_punctuator_model(metrics)
        tokenizer, target_token2id, id2target = load_tokenization_model()
        flag = True

def instruct_msg():
    reply_msg = "enter Russian text without punctuation."
    return reply_msg
def chatbot_response(user_msg):
    print(user_msg)
    reply_msg = tokenization(user_msg, tokenizer, target_token2id, id2target, config, model, device)
    return reply_msg

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    if userText == "!help":
        return instruct_msg()
    else:
        return chatbot_response(userText)


if __name__ == "__main__":
    ready_model()
    print("Model is ready! Use command `help` to know my ability.")

    app.run()


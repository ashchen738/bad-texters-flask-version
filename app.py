import flask
from flask import Flask, render_template, url_for, request
import os
from werkzeug.utils import secure_filename
import backend

app = flask.Flask(__name__)
currpath = os.path.abspath(os.curdir)
app.config['UPLOAD_FOLDER'] = currpath


@app.route('/', methods=['GET'])
def complete_request():
    return render_template('index.jinja2')


@app.route('/', methods=['POST'])
def return_and_save_file():
    print("works here?")
    print(request.files)
    print('*')
    uploaded = request.files['file']
    print(uploaded.filename)
    if len(uploaded.filename) > 0:
        finame = secure_filename(uploaded.filename)
        uploaded.save(os.path.join(app.config['UPLOAD_FOLDER'], finame))

    li, recipient, sender = backend.read_file(uploaded.filename)
    print(li)
    scores = backend.calc_score(li, recipient, sender)
    sub_scores1, sub_scores2 = backend.compile_all(li, recipient, sender)
    sub_scores1["overall_score"] = scores[0]
    sub_scores2["overall_score"] = scores[1]
    # return sub_scores1
    return render_template("output.jinja2", senderdict=sub_scores1, recipdict=sub_scores2)
    # {"overall_score":score, "common_words":list of words (choose the first one from this list),
    # "left_on_read": number, "react":number (time)), "sentiment":number(0-1), "texting_dif":number of
    # text sent and received, "avg_length":length of texts, "num_reactions":numberofreactions}


app.run()

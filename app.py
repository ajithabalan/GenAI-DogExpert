import os
from flask import Flask, request, render_template
from model import predict_breed
from chatbot import chat_with_dog_ai  
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
   
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            breed = predict_breed(filepath)
            return render_template("index.html", filename=filename, breed=breed)
    return render_template("index.html")
    
@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        user_query = request.form["query"]
        response = chat_with_dog_ai(user_query)
        return render_template("chat.html", query=user_query, response=response)
    
    return render_template("chat.html", query=None, response=None)



if __name__ == "__main__":
    app.run(debug=True)

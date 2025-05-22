from flask import Flask, render_template, request
from app.fake_news_detector import detect_fake_news
from app.deepfake_detector import detect_deepfake

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        if 'news_text' in request.form:
            news_text = request.form['news_text']
            result = detect_fake_news(news_text)
        elif 'deepfake_image' in request.files:
            image = request.files['deepfake_image']
            result = detect_deepfake(image)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///speeches.db'  # 3 / για relative path
db = SQLAlchemy(app)



class Speeches(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    member_name = db.Column(db.String(100), nullable=False)
    sitting_date = db.Column(db.DateTime, nullable=False)
    # parliamentary_period = db.Column(db.String(50), nullable=False)
    # parliamentary_session = db.Column(db.String(50), nullable=False)
    # parliamentary_sitting = db.Column(db.String(50), nullable=False)
    political_party = db.Column(db.String(100), nullable=False)
    # government = db.Column(db.String(100), nullable=False)
    # member_region = db.Column(db.String(100), nullable=False)
    # roles = db.Column(db.String(100), nullable=False)
    # member_gender = db.Column(db.String(10), nullable=False)
    speech = db.Column(db.String(500), nullable=False, default='')
      

    # def __repr__(self):
    #     return f'<Speech {self.id} of {self.member_name}> on {self.sitting_date}'
    def __repr__(self):
        return f'<Speech {self.id}'
    


    
# Search Engine For Speeches using Flask and SQLAlchemy
@app.route('/', methods=['POST', 'GET'])  # για να μπορώ και να στείλω και να παρω δεδομένα
def index():
    if request.method == 'POST':  # ανάδραση της σελίδας σε δεδομένα που στέλνει ο χρήστης μέσω της φόρμας
        query = request.form['search']  # αντιστοιχει στο name του input στην html φόρμα
        # print(query)
        try:
            # Εδώ θα έχω έτοια τα ids που θέλω και το filter θα γινεται μόνο με το id
            speeches = db.session.query(Speeches).filter(Speeches.speech.contains(query)).all()
            return render_template('index.html', speeches=speeches)
        except:
            return 'There was an issue searching the database'

    else:
        return render_template('index.html')

@app.route('/read/<int:id>')
def read(id):  # ξεχωριστή σελίδα για κάθε ομιλία όταν πατάει ο χρήστης το read speech
    speech = Speeches.query.get_or_404(id)

    return render_template('read.html', speech=speech)


if __name__ == '__main__':
    with app.app_context():
        db.create_all()

        # test_speech = Speeches(
        #     member_name="John Doe",
        #     sitting_date=datetime.strptime("2023-10-01", "%Y-%m-%d"),
        #     political_party="Independent",
        #     speech="This is a test speech."
        # )

        

        # speech_to_delete = Speeches.query.get(4)
        # if speech_to_delete:
        #     db.session.delete(speech_to_delete)
        #     db.session.commit()
        
       
    app.run(debug=True)
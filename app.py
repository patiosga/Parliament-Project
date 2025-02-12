from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pandas as pd
from top_k_query import top_k_query
from preprocess_speech import preprocess_speech

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///speeches.db'  # 3 / για relative path
db = SQLAlchemy(app)



class Speeches(db.Model):
    id = db.Column(db.Integer, primary_key=True, nullable=False)
    member_name = db.Column(db.String(100), nullable=False)
    sitting_date = db.Column(db.DateTime, nullable=False)
    political_party = db.Column(db.String(100), nullable=False)
    speech = db.Column(db.String(500), nullable=False, default='')
      
    def __repr__(self):
        return f'<Speech {self.id}'
    

# Load DataFrame and Insert into Database
def load_dataframe_to_db(df: pd.DataFrame):
    with app.app_context():
        db.create_all()  # Ensure tables exist

        # Convert DataFrame to Dictionary format
        for _, row in df.iterrows():
            speech = Speeches(
                id=row['id'],  # Set ID manually
                member_name=row['member_name'],
                sitting_date=datetime.strptime(row['sitting_date'], "%d/%m/%Y"),
                political_party=row['political_party'],
                speech=row['speech']
            )

            if db.session.get(Speeches, row['id']):
                print(f"Speech with ID {row['id']} already exists. Skipping...")
                continue
            else:
                db.session.add(speech)

        db.session.commit()
        print("Data successfully inserted!")
    


    
# Search Engine For Speeches using Flask and SQLAlchemy
@app.route('/', methods=['POST', 'GET'])  # για να μπορώ και να στείλω και να παρω δεδομένα
def index():
    if request.method == 'POST':  # ανάδραση της σελίδας σε δεδομένα που στέλνει ο χρήστης μέσω της φόρμας
        query = request.form['search']  # αντιστοιχει στο name του input στην html φόρμα
        # print(query)
        try:
            # Process the query (stemming, stop words removal, etc.)
            processed_query = preprocess_speech(query)
            # Split into terms
            processed_query_terms = processed_query.split()
            # Retrieve top-k document IDs
            speech_ids = top_k_query(processed_query_terms, 20)
            # print(speech_ids)
            # Return the speeches in the same order as the IDs (the ids are in the order of relevance)
            speeches = db.session.query(Speeches).filter(Speeches.id.in_(speech_ids)).all()
            speeches_dict = {speech.id: speech for speech in speeches}
            speeches = [speeches_dict[id] for id in speech_ids if id in speeches_dict]
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

    load = False
    if load:
        # Load the DataFrame
        df = pd.read_csv('data.csv')
        # Load data into the database
        load_dataframe_to_db(df)
        print("Data loaded into database.")
    else:

        with app.app_context():
            db.create_all()

        # test_speech = Speeches(
        #     member_name="John Doe",
        #     sitting_date=datetime.strptime("2023-10-01", "%Y-%m-%d"),
        #     political_party="Independent",
        #     speech="This is a test speech."
        # )
        # db.session.add(test_speech)
        # db.session.commit()


        

        # speech_to_delete = Speeches.query.get(4)
        # if speech_to_delete:
        #     db.session.delete(speech_to_delete)
        #     db.session.commit()
        
       
    app.run(debug=True)
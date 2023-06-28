import pyrebase
firebaseConfig = {
  "apiKey": "AIzaSyD9LcPsV7StilRfvnHMtHAWGSiPmaYX7Ic",
  "authDomain": "social-distancing-4791d.firebaseapp.com",
  "databaseURL": "https://social-distancing-4791d-default-rtdb.firebaseio.com",
  "projectId": "social-distancing-4791d",
  "storageBucket": "social-distancing-4791d.appspot.com",
  "messagingSenderId": "657450370621",
  "appId": "1:657450370621:web:4c2d5351cfd49481b74c54",
  "measurementId": "G-NCHBRSC8W4"
};
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()


# Log the user in anonymously
#user = auth.sign_in_anonymous()

# Get a reference to the database service
db = firebase.database()

# data to save
data = {
    "name": "Mortimer 'Morty' Smith"
}

# Pass the user's idToken to the push method
results = db.child("users").child('dhurkesh').set(data)

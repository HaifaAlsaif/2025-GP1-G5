import firebase_admin
from firebase_admin import credentials, firestore, db

cred = credentials.Certificate("service-account.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        "projectId": "trustlens-e9038",
        "databaseURL": "https://trustlens-e9038-default-rtdb.firebaseio.com"
    })

# Firestore client
db = firestore.client()


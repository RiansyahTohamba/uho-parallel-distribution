import jwt
from flask import Flask, request, jsonify
from flask_cors import CORS

from functools import wraps
import json
import os
from datetime import datetime, timedelta

SECRET_KEY = os.environ.get("SECRET_KEY", "rahasia_super_aman")  # Use environment variable
app = Flask(__name__)
CORS(app)  # Enable CORS for Svelte frontend

DATA_FILE = "db_participants.json"

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]

        if not token:
            return jsonify({"message": "Token tidak ditemukan"}), 401

        try:
            decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            request.user = decoded  # Make decoded token available in route
        except jwt.ExpiredSignatureError:
            return jsonify({"message": "Token kedaluwarsa"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"message": "Token tidak valid"}), 401

        return f(*args, **kwargs)

    return decorated

def read_data():
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def write_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

def is_user_registered(username, password):
    return username == "admin" and password == "password"

# Login endpoint to generate token
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    
    # Simple authentication (replace with real authentication)
    if is_user_registered(username, password):
        token = jwt.encode({
            "user_id": username,
            "exp": datetime.utcnow() + timedelta(hours=24)
        }, SECRET_KEY, algorithm="HS256")
        
        return jsonify({"token": token, "message": "Login berhasil"}), 200
    
    return jsonify({"message": "Username atau password salah"}), 401



@app.route("/participants", methods=["GET"])
@token_required
def get_participants():
    data = read_data()
    return jsonify(data), 200

@app.route("/participants", methods=["POST"])
@token_required
def add_participant():
    new_participant = request.get_json()
    data = read_data()
    
    # Generate ID if not provided
    if "id" not in new_participant:
        new_participant["id"] = len(data) + 1
    
    data.append(new_participant)
    write_data(data)
    
    return jsonify({"message": "Participant added", "data": new_participant}), 201

@app.route("/participants/<int:participant_id>", methods=["PUT"])
@token_required
def update_participant(participant_id):
    updated_data = request.get_json()
    data = read_data()
    
    for i, participant in enumerate(data):
        if participant.get("id") == participant_id:
            data[i] = {**participant, **updated_data, "id": participant_id}
            write_data(data)
            return jsonify({"message": "Participant updated", "data": data[i]}), 200
    
    return jsonify({"message": "Participant not found"}), 404

@app.route("/participants/<int:participant_id>", methods=["DELETE"])
@token_required
def delete_participant(participant_id):
    data = read_data()
    
    for i, participant in enumerate(data):
        if participant.get("id") == participant_id:
            deleted = data.pop(i)
            write_data(data)
            return jsonify({"message": "Participant deleted", "data": deleted}), 200
    
    return jsonify({"message": "Participant not found"}), 404

if __name__ == "__main__":
    app.run(debug=True, port=55110)
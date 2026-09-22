from flask import Flask, jsonify
from database import init_db, get_eligible_maba

app = Flask(__name__)

@app.route('/camaba-eligible', methods=['GET'])
def camaba_eligible():
    try:
        maba_list = get_eligible_maba()
        return jsonify({
            'status': 'success',
            'data': maba_list,
            'count': len(maba_list)
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    init_db()
    print("Pendaftaran Maba Service running on http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)

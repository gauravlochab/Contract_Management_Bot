
"""Python file to serve as the frontend"""
import os
import time
from flask import Flask, request, jsonify
import json
import logging
logging.basicConfig(level=logging.INFO)
from vectorization import *
app = Flask(__name__)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename=file.filename
        # Upload to S3
        #print(file.getvalue())
        try:
            print('lolololololol')
            s3_file_url = upload_file_to_s3(file, bucket_name,filename)
            print('s3_file_url',s3_file_url)
            # Assuming the process_pdf function is defined elsewhere to process the PDF file.
            return jsonify({'message': 'File uploaded successfully', 'url': s3_file_url}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Unsupported file type'}), 400

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        #print(file.getvalue())
        filename=file.filename
        logging.info("filename: %s", filename)
        try:
            text_data = process_file(file)
            #print(text_data)
            response_json = get_response_openai(text_data)
            upload_to_mongodb(response_json,filename)
            #print(response_json)
            return json.loads(response_json[0])
        except Exception as e:
            logging.error("Error processing file: %s", str(e))
            return jsonify({'error': 'Failed to process file: ' + str(e)}), 500
    else:
        return jsonify({'error': 'Unsupported file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=8000)

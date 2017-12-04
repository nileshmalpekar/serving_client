from __future__ import print_function
# http://flask.pocoo.org/docs/patterns/fileuploads/
import os
from flask import Flask, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

host = 'tensorflow-serving'
port = '9000'
channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

def allowed_file(filename):
  # this has changed from the original example because the original did not work for me
  name, ext = os.path.splitext(filename)
  return ext[1:].lower() in ALLOWED_EXTENSIONS
  # return filename[-4:].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
  if request.method == 'POST':
    file = request.files['file']
    if file and allowed_file(file.filename):
      filename = secure_filename(file.filename)
      # See prediction_service.proto for gRPC request/response details.
      data = file.read()
      req = predict_pb2.PredictRequest()
      req.model_spec.name = 'inception'
      req.model_spec.signature_name = 'predict_images'
      req.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(data, shape=[1]))
      result = stub.Predict(req, 10.0)  # 10 secs timeout
      return repr(result.outputs['classes'].string_val[0])

  return '''
  <!doctype html>
    <title>Upload new File</title>
      <h1>Upload new File</h1>
      <form action="" method=post enctype=multipart/form-data>
        <p>
          <input type=file name=file>
          <input type=submit value=Upload>
      </form>
'''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
  return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=80, debug=True)


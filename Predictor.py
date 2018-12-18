from tensorflow.contrib import predictor
from TFRecordWriter import load_image

img_path = '/home/aicore/Projects/Dashboard/icons/single/1.jpg'

predict_fn = predictor.from_saved_model(export_dir="cnn_model")
predictions = predict_fn(
    {"test/image": load_image(img_path)})
print(predictions)

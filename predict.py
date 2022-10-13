import tensorflow as tf
import os
import time
import tarfile
from google.cloud import storage

class Prediction():
    def __init__(self, model_name, model_type, input_shape, verbose=False):
        self.model_name = model_name
        self.model_type = model_type
        self.input_shape = input_shape
        self.verbose = verbose
        self.model_path = f'/tmp/{self.model_name}.tar.gz'
        self.model = None

    def _prep_image(self, cloud_img_path):
        target_size = (self.input_shape, self.input_shape)
        img = tf.keras.utils.load_img(
            path=cloud_img_path,
            target_size=target_size
            )
        return tf.keras.utils.img_to_array(img)

    def _download_model(self):
        """Downloads model from cloud storage."""
        storage_client = storage.Client()
        bucket = storage_client.bucket('coffeebot')
        blob_name = os.path.join(
            f'image_classifiers/{self.model_type}/',
            f'{self.model_name}.tar.gz'
            )
        blob = bucket.blob(blob_name)        
        blob.download_to_filename(self.model_path)

    def _check_local_model(self):
        in_tmp = os.path.isdir(f'/tmp/{self.model_name}')
        in_private_tmp = os.path.isdir(f'/private/tmp/{self.model_name}')
        return in_tmp or in_private_tmp

    def _load_model(self):
        # Get from cloud storage
        # Check if local model exists
        is_local = self._check_local_model()
        if not is_local:
            if self.verbose:
                print('Downloading model from cloud storage...')
            self._download_model()
            # Untar
            with tarfile.open(self.model_path) as t:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(t, "/tmp")
        self.model_path = self.model_path.split('.')[0]
        # Load
        if self.verbose:
            print('Loading model into memory...')
        self.model = tf.keras.models.load_model(self.model_path)

    def predict(self, tensors):
        tensors = tf.stack(tensors)
        start = time.time()
        predictions = self.model.predict(tensors)
        end = time.time()
        self.prediction_time = end-start
        return predictions

    def remote_predict(self, img_paths):
        if not self.model:
            self._load_model()
        tensors = []
        for path in img_paths:
            if self.verbose:
                print(f'Prepping image: {path}')
            tensors.append(self._prep_image(cloud_img_path=path))
        self.predictions = self.predict(tensors)


if __name__ == '__main__':
    predictor = Prediction(
        model_name='EfficientNetB3_1643388370',
        input_shape=512,
        model_type='cats_v_dogs',
        verbose=True
    )

    cat_dir = 'data/samples/cats'
    dog_dir = 'data/samples/dogs'
    for img_path in os.listdir(cat_dir):
        predictor.remote_predict([os.path.join(cat_dir, img_path)])
        print(predictor.predictions)

    for img_path in os.listdir(dog_dir):
        predictor.remote_predict([os.path.join(dog_dir, img_path)])
        print(predictor.predictions)

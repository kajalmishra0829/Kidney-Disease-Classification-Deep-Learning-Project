import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    
    def predict(self):
        # load model
        model = load_model(os.path.join("model", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = test_image/255.0
        test_image = np.expand_dims(test_image, axis = 0)
        raw_output = model.predict(test_image)
        result = np.argmax(raw_output, axis=1)

        # --- detailed logs ---
        print(f"Raw model output (probabilities): {raw_output}")
        print(f"Normal probability : {raw_output[0][0]*100:.2f}%")
        print(f"Tumor probability  : {raw_output[0][1]*100:.2f}%")
        print(f"Predicted class index: {result[0]}")
        print(f"Predicted label: {'Tumor' if result[0] == 1 else 'Normal'}")
        # ---------------------


        if result[0] == 1:
            prediction = 'Tumor'
            return [{ 
                "image": prediction, 
                "Tumor probability": f"{raw_output[0][1]*100:.2f}%"
            }]
        else:
            prediction = 'Normal'
        return [{ 
            "image": prediction,
            "Normal probability": f"{raw_output[0][0]*100:.2f}%",
        }]
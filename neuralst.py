import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg19
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import requests
from PIL import Image
from io import BytesIO

def download_and_process_image(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).resize((400, 400))  # Ensure consistent size
    img = img_to_array(image)
    img = np.expand_dims(img, axis=0)
    return tf.convert_to_tensor(vgg19.preprocess_input(img), dtype=tf.float32)

def deprocess_image(img):
    img = img.reshape((400, 400, 3))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')
    return img

base_model = vgg19.VGG19(weights='imagenet', include_top=False, input_shape=(400, 400, 3))  # Explicitly set input shape

feature_extractor = Model(inputs=base_model.input, outputs=[base_model.get_layer(name).output for name in ['block5_conv2', 'block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']])

def get_feature_representations(content_url, style_url):
    content_image = download_and_process_image(content_url)
    style_image = download_and_process_image(style_url)

    content_features = feature_extractor(content_image)[0]  # Extract content feature correctly
    style_features = feature_extractor(style_image)  # Ensure style features match layer count

    return content_features, style_features



def compute_content_loss(content, generated):
    return tf.reduce_mean(tf.square(content - generated))

def gram_matrix(tensor):
    channels = int(tensor.shape[-1])
    vectorized = tf.reshape(tensor, [-1, channels])
    return tf.matmul(tf.transpose(vectorized), vectorized)

def compute_style_loss(style, generated):
    return tf.reduce_mean(tf.square(gram_matrix(style) - gram_matrix(generated)))

def total_loss(content_weight, style_weight, content_features, style_features, generated_features):
    content_loss = compute_content_loss(content_features[0], generated_features[0])

    # Ensure we use the correct layers for style loss computation
    style_loss = sum(compute_style_loss(s, g) for s, g in zip(style_features, generated_features))

    return content_weight * content_loss + style_weight * style_loss



def neural_style_transfer(content_url, style_url, iterations=100, content_weight=1e4, style_weight=1e-2):
    content_features, style_features = get_feature_representations(content_url, style_url)
    generated_image = tf.Variable(download_and_process_image(content_url), dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5.0)
    for i in range(iterations):
        with tf.GradientTape() as tape:
            generated_features = feature_extractor(generated_image)
            loss = total_loss(content_weight, style_weight, content_features, style_features, generated_features)
        gradients = tape.gradient(loss, generated_image)
        optimizer.apply_gradients([(gradients, generated_image)])
        generated_image.assign(tf.clip_by_value(generated_image, -103.939, 255 - 103.939))
        if i % 20 == 0:
            print(f"Iteration {i}, Loss: {loss.numpy()}")
    return deprocess_image(generated_image.numpy())

def main():
    content_url = input("Enter content image URL: ")
    style_url = input("Enter style image URL: ")
    result = neural_style_transfer(content_url, style_url)
    plt.imshow(result)
    plt.axis("off")
    plt.show()
    cv2.imwrite("styled_image.jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    main()

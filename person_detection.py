import numpy as np
import argparse
import tensorflow as tf
import cv2
import pathlib
import os
import pandas as pd
from PIL import Image
import datetime

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def run_inference(model, category_index, cap, show_video_steam, label_to_look_for, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(output_directory + '/images', exist_ok=True)

    if os.path.exists(output_directory + '/results.csv'):
        df = pd.read_csv(output_directory + '/results.csv')
    else:
        df = pd.DataFrame(columns=['timestamp', 'img_path'])

    while True:
        ret, image_np = cap.read()
        # Copy image for later
        image_show = np.copy(image_np)
        image_height, image_width, _ = image_np.shape
        # Actual detection.
        output_dict = run_inference_for_single_image(model, image_np)
        # Visualization of the results of a detection.
        if show_video_steam:
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks_reframed', None),
                use_normalized_coordinates=True,
                line_thickness=8)
            cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

        # Get data(label, xmin, ymin, xmax, ymax)
        output = []
        for index, score in enumerate(output_dict['detection_scores']):
            label = category_index[output_dict['detection_classes'][index]]['name']
            ymin, xmin, ymax, xmax = output_dict['detection_boxes'][index]
            output.append((label, int(xmin * image_width), int(ymin * image_height), int(xmax * image_width),
                            int(ymax * image_height)))

            # Save incident (could be extended to send a email or something)
            for l, x_min, y_min, x_max, y_max in output:
                if l == label_to_look_for:
                    array = cv2.cvtColor(np.array(image_show), cv2.COLOR_RGB2BGR)
                    image = Image.fromarray(array)
                    cropped_img = image.crop((x_min, y_min, x_max, y_max))
                    file_path = output_directory + '/images/' + str(len(df)) + '.jpg'
                    cropped_img.save(file_path, "JPEG", icc_profile=cropped_img.info.get('icc_profile'))
                    df.loc[len(df)] = [datetime.datetime.now(), file_path]
                    df.to_csv(output_directory + '/results.csv', index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects inside webcam videostream')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model Path')
    parser.add_argument('-l', '--labelmap', type=str, required=True, help='Path to Labelmap')
    parser.add_argument('-s', '--show', default=True, action='store_true', help='Show window')
    parser.add_argument('-la', '--label', default='person', type=str, help='Label name to detect')
    parser.add_argument('-o', '--output_directory', default='results', type=str, help='Directory for the outputs')
    args = parser.parse_args()

    detection_model = load_model(args.model)
    category_index = label_map_util.create_category_index_from_labelmap(args.labelmap, use_display_name=True)

    cap = cv2.VideoCapture(0)
    run_inference(detection_model, category_index, cap, args.show, args.label, args.output_directory)

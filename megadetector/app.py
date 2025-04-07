import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np

from PytorchWildlife.models import detection as pw_detection

detection_model = pw_detection.MegaDetectorV6(version="MDV6-yolov9-c")


def load_example_images(folder_path="./photos"):
    examples = [
        Image.open(os.path.join(folder_path, p)) for p in os.listdir(folder_path)
    ]
    return examples


example_images = load_example_images()


def draw_info(draw, image, bbox, label):
    width, height = image.size
    border_size, font_size = int(width * 0.01), int(width * 0.05)

    width, height = image.size
    im_font = ImageFont.truetype("ArialUnicode.ttf", font_size)

    xmin, ymin, xmax, ymax = bbox[0]
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=border_size)
    draw.text((xmin, ymin - 1.5 * font_size), label, fill="red", font=im_font)


def classifier(image):
    """Use MegaDetector to detect animals in images. Returns
    new image with detection results."""
    image_array = np.array(image.convert("RGB"))
    result = detection_model.single_image_detection(image_array)
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    for bbox, label in zip(result["detections"], result["labels"]):
        draw_info(draw, image, bbox, label)

    return img_with_boxes


# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## MegaDetector Example")

    with gr.Row():
        input_image = gr.Image(label="Input Image", type="pil")
        output_image = gr.Image(label="Classified Image", type="pil")

    classify_button = gr.Button("Classify")
    classify_button.click(fn=classifier, inputs=input_image, outputs=output_image)

    gr.Examples(
        examples=example_images, inputs=input_image, label="Choose an example image"
    )

demo.launch()

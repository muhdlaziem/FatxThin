import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

from PIL import  Image
from io import  BytesIO
import base64
import numpy as np
from utils import load_model, align_images, move_and_show, project_image    

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

print('Loading Models...')
proj, generator, landmarks_detector = load_model()
fatness_direction = np.load('stylegan2encoder/directions/fatness_direction.npy')
print('Models Loaded...')

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(img):
    output_buffer = BytesIO()
    img.save(output_buffer, format='PNG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str.decode()

@app.route('/')
def upload_form():
	return render_template('home.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
        
    if file and allowed_file(file.filename):
        # img = Image.open(BytesIO(file.read()))
        im = align_images(BytesIO(file.read()), landmarks_detector)
        temp_folder = hashlib.md5(im[0].tobytes()).hexdigest()
        latent = project_image(proj, im[0], tmp_dir=temp_folder)
        flash('Image successfully encoded and displayed below')

        yield render_template('home.html', image="data:image/png;base64," + image_to_base64(generate_image(latent, generator)))


        transform = move_and_show(latent, fatness_direction, 0.5, generator)
        flash('Image successfully transformed and displayed below')

        return render_template('home.html', result="data:image/png;base64," + image_to_base64(transform))
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)



if __name__ == "__main__":
    app.run(port=8080,debug=True)

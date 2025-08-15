import os
import tempfile
import magic
from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg'}
# Define the path for the upload folder relative to the app's root path
UPLOAD_FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), UPLOAD_FOLDER)


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER_PATH
app.secret_key = 'super secret key'

# Ensure the upload folder exists and is not executable
if not os.path.exists(UPLOAD_FOLDER_PATH):
    os.makedirs(UPLOAD_FOLDER_PATH)
os.chmod(UPLOAD_FOLDER_PATH, 0o755) # Read and write for owner, read for others


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_mime_type(file_stream):
    """
    Read the file's magic number to determine the mime type
    """
    # Read the first 2048 bytes to identify the file type
    file_header = file_stream.read(2048)
    file_stream.seek(0)  # Reset stream position
    mime_type = magic.from_buffer(file_header, mime=True)
    return mime_type

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # 1. File Path Validation
            filename = secure_filename(file.filename)

            # 2. File Type Validation (Magic Number)
            mime_type = get_file_mime_type(file.stream)
            if mime_type not in ['image/png', 'image/jpeg']:
                flash('Invalid file type. Only PNG and JPG are allowed.')
                return redirect(request.url)

            # 3. Temporary File Management
            try:
                # Save the file to a secure temporary file first
                with tempfile.NamedTemporaryFile(delete=False, dir=app.config['UPLOAD_FOLDER'], suffix=".tmp") as temp_file:
                    file.save(temp_file.name)
                    temp_file_path = temp_file.name

                # Once validated, move the temporary file to the final destination
                final_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                os.rename(temp_file_path, final_path)

                flash('File successfully uploaded')
                return redirect(url_for('upload_file'))
            except Exception as e:
                flash(f'An error occurred: {e}')
                # Clean up the temporary file if it exists
                if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                return redirect(request.url)

        else:
            flash('Allowed file types are png, jpg')
            return redirect(request.url)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

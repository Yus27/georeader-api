import os
from flask import Flask, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['rdr', 'txt', 'gpr', 'gpr2', 'rd3', 'dzt', 'sgy'])
UPLOAD_FOLDER = 'tmp'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method != 'POST':
        return jsonify({"Error": "Неверный метод"})

    # print(request.files)
    if 'file' not in request.files:
        return jsonify({"Error": "Файл не передан1"})
    file = request.files['file']
    if not file:
        return jsonify({"Error": "Файл не передан2"})
    if not allowed_file(file.filename):
        return jsonify({"Error": "Неверный формат файла"})

    ext = os.path.splitext(file.filename)[1]
    filename = secure_filename(file.filename) + ext
    filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(filename)
        ext = os.path.splitext(filename)[1]

        Rad = {"Data": None, "Stage": 1.0, "TimeBase": 512, "AntDist": 0, "DefaultV": 0.1, "GPRUnit": "", "AntenName": "","Frequency": 1000, "Error": None}

        if ext == ".txt":
            from import_data import importFromTXT
            res = importFromTXT(filename)
            if res is not None:
                Rad["Data"] = res
                Rad["GPRUnit"] = "ЛОЗА"
            else:
                return jsonify({"Error": "Не удалось загрузить файл"})
        elif ext == ".gpr" or ext == ".gpr2":
            isOldVersion = ext == ".gpr"
            from import_data import importFromGeoScan
            res = importFromGeoScan(filename, IsOldVersion=isOldVersion)
            if res is not None:
                Rad["Data"], Rad["Stage"], Rad["TimeBase"], Rad["AntDist"], Rad["DefaultV"], _, _, _, _, _, _, _, \
                Rad["GPRUnit"], Rad["AntenName"], Rad["Frequency"], _ = res
            else:
                return jsonify({"Error": "Не удалось загрузить файл"})

        Rad["Data"] = Rad["Data"].tolist()
        return jsonify(Rad)
    finally:
        pass
        # os.remove(filename)


# if __name__ == '__main__':
#     app.run(debug=True)
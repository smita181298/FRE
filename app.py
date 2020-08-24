import os
import cv2
import time
import tensorflow as tf
from keras.models import load_model
import numpy
from numpy import asarray, load, expand_dims
from random import choice
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle
from PIL import Image
from mtcnn.mtcnn import MTCNN
from flask import Flask, jsonify, request, render_template, Response
from flask_cors import CORS
import warnings
import skvideo.io
from imutils.video import VideoStream
import imutils
import json
from json import JSONEncoder

# from .run_model import deblur1, contrast, dark, blur, convertContrastHigh, convertContrastLow, convertdark
import run_model as runmodel
global graph
graph = tf.compat.v1.get_default_graph()
warnings.filterwarnings('ignore')
app = Flask(__name__)
CORS(app)
# json = FlaskJSON(app)
detector = MTCNN()
height1 = 12
width1 = 12

IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
VIDEO_EXTENSIONS = {'m4v', 'mp4', 'mov', 'webm'}
with graph.as_default():
    model = load_model('models/facenet_keras.h5')


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def imageQuality(path):
    contrast1 = runmodel.contrast(path)
    dark1 = runmodel.dark(path)
    blur1 = runmodel.blur(path)
    print(contrast1, dark1, blur1)
    if dark1 == "dark":
        path2 = runmodel.convertdark(path)
    elif blur1 == "blurry":
        # print("asdfghjk")
        path2 = runmodel.deblur1(path)
    elif contrast1 == "high contrast":
        path2 = runmodel.convertContrastHigh(path)
    elif contrast1 == "low contrast":
        path2 = runmodel.convertContrastLow(path)
    else:
        path2 = path
    return path2


@app.route('/')
def upload_file():
    return render_template("upload.html")


def draw_image_with_boxes(filename, result_list, label):
    global height1, width1
    height1, width1, layers = filename.shape
    size = (width1, height1)
    frame_array = filename
    for result in result_list:
        x, y, width, height = result['box']
        cv2.rectangle(filename, (x, y), (x+width, y+height), (0, 255, 0), 2)
        cv2.putText(filename, label, (x-width, y),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 255, 155), 1)
    return frame_array


def get_embedding(face_pixels):
    with graph.as_default():
        face_pixels = face_pixels.astype('float32')
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        samples = expand_dims(face_pixels, axis=0)
        yhat = model.predict(samples)
    return yhat[0]


def extract_face(filename, required_size=(160, 160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    # print("gfys")
    with graph.as_default():
        results = detector.detect_faces(pixels)
    # print("yg")
    faces = list()
    images = list()
    for k in range(len(results)):
        x1, y1, width, height = results[k]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        images.append(image)
        face_array = asarray(image)
        faces.append(face_array)
    return faces, images, results


@app.route('/extractfaces', methods=['GET', 'POST'])
def load_faces():
    try:
        with open('db/extract.p', 'rb') as fp:
            extract = pickle.load(fp)
    except:
        extract = []
        with open('db/extract.p', 'wb') as fp:
            pickle.dump(extract, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # f = request.files['file']
    Files = request.files.getlist("files")
    # f = Files[0]
    ID = request.form.get("personID")
    # print(f.filename)
    imagelist = list()
    num = list()
    F = Files[0]
    if F.filename.rsplit('.', 1)[1].lower() in IMAGE_EXTENSIONS:
        for f in Files:
            if f.filename.rsplit('.', 1)[1].lower() in IMAGE_EXTENSIONS:
                f.save("static/images/"+f.filename)
                path = "static/images/"+f.filename
                path2 = imageQuality(path)
                print(path2)
                path = path2
                face, images, results = extract_face(path)

                if(len(face) > 0):
                    for i in range(len(images)):
                        path = "static/extract/extracted_" + \
                            ID+str(time.time())+".png"
                        image = images[i].save(path)
                        num.append(asarray(images[i]))
                        imagelist.append(path)
                        extract.append(images[i])
                    if(len(face) == 1):
                        h = "face found"
                        with open('db/extract.p', 'wb') as fp:
                            pickle.dump(
                                extract, fp, protocol=pickle.HIGHEST_PROTOCOL)

                    elif(len(face) > 1):
                        os.remove(os.path.join("static/images/", f.filename))
                        h = "multiple faces found"
                else:
                    h = "no face found"
                response = json.dumps(
                    h)+str(app.response_class(response=json.dumps(h), status=200, mimetype='application/json'))
                print(response)
                # return jsonify(response)
        Return = {"imagelist": imagelist, "array": num}
        encodedNumpyData = json.dumps(returnurn, cls=NumpyArrayEncoder)
        return Response(encodedNumpyData)
        # return render_template("extract.html", results=imagelist, heading=response)
    else:
        imagelist = list()
        paths = list()
        num = list()
        for f in Files:
            if f.filename.rsplit('.', 1)[1].lower() in VIDEO_EXTENSIONS:
                f.save("static/videos/"+ID+"." +
                       f.filename.rsplit('.', 1)[1].lower())
                vidcap = cv2.VideoCapture(
                    "static/videos/"+ID+"."+f.filename.rsplit('.', 1)[1].lower())
                success, image = vidcap.read()
                count = 0
                X = list()
                m = list()
                frame_array = list()
                frame_width = int(vidcap.get(3))
                frame_height = int(vidcap.get(4))
                size = (frame_width, frame_height)
                print(size)
                pathvideo = "static/videos/"+ID+str(time.time())+".mp4"
                result = cv2.VideoWriter(pathvideo,
                                         cv2.VideoWriter_fourcc(*'MJPG'),
                                         10, size)
                paths.append(pathvideo)
                while success:
                    print("helloooo")
                    success, image = vidcap.read()

                    try:
                        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        path = "static/images/"+ID+"_"+str(time.time())+".png"
                        cv2.imwrite(path, image)
                        path2 = imageQuality(path)
                        print(path2)
                        path = path2

                        face, images, results = extract_face(path)
                        label = None
                        file = draw_image_with_boxes(image, results, label)
          
                        x = result.write(file)
                        print(x)
                        frame_array.append(file)
                    except:
                        face = []
                    print(frame_array, "framearray")

                    if len(face) > 1:
                        m.append(2)
                        for i in range(len(images)):
                            path = "static/exvideos/"+ID + \
                                "_"+str(time.time())+".png"

                            image = images[i].save(path)
                            imagelist.append(path)
                            num.append(asarray(images[i]))
                            extract.append(images[i])
                        with open('db/extract.p', 'wb') as fp:
                            pickle.dump(
                                extract, fp, protocol=pickle.HIGHEST_PROTOCOL)

                    elif len(face) < 1:

                        m.append(0)
                    else:
                        path = "static/exvideos/"+ID + \
                            "_"+str(time.time())+".png"
                        image = images[0].save(path)
                        imagelist.append(path)
                        num.append(asarray(images[0]))

                        m.append(1)
                        extract.append(images[0])
                        with open('db/extract.p', 'wb') as fp:
                            pickle.dump(
                                extract, fp, protocol=pickle.HIGHEST_PROTOCOL)
                result.release()
                if 2 in m:
                    h = "Multiple faces found."
                elif 1 in m:
                    h = "One face found"
                else:
                    h = "No faces found."
        print(pathvideo, "path")
        Return = {"imagepath": imagelist, "array": num, "heading": h}
        encodedNumpyData = json.dumps(Return, cls=NumpyArrayEncoder)
        return Response(encodedNumpyData)
        # return render_template("extractvideos.html", results=imagelist, videoname=pathvideo, heading=h)


@app.route('/countfaces', methods=['GET', 'POST'])
def countfaces():
    path = ""
    if request.method == 'POST':
        clicked = request.get_json()
        print("clicked", clicked)
        print(request.files)
        ID = request.form.get("personID")
        print("ID", ID)
        Files = request.files.getlist("files")
        paths = list()
        N = 0
        num = list()
        for f in Files:
            print("f", f)
            f.save("static/count/"+f.filename)
            path = "static/count/" + f.filename
            path2 = imageQuality(path)
            print(path2)
            path = path2
            paths.append(path)
            image = Image.open(path)
            image = image.convert('RGB')
            pixels = asarray(image)
            with graph.as_default():
                faces = detector.detect_faces(pixels)
            data = cv2.imread(path)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            i = len(faces)
            h = "number of faces : "+str(i)
            j = list()
            j.append(i)
            N = N+i
            for result in faces:
                x, y, width, height = result['box']
                cv2.rectangle(data, (x, y), (x+width, y+height),
                              (0, 255, 0), 2)
                image = Image.fromarray(data)
                image = image.save(path)
            num.append(data)

        data = {"number of faces": N}
        # response = json.dumps(data)+str(app.response_class(
        #     response=json.dumps(data), status=200, mimetype='application/json'))
        Return = {"number of faces": N, "path": paths, "array": num}
        encodedNumpyData = json.dumps(Return, cls=NumpyArrayEncoder)
        return Response(encodedNumpyData)
        # return render_template("count.html", results=paths, heading=response)


@app.route('/addface', methods=['GET', 'POST'])
def addface():
    try:
        with open('db/people.p', 'rb') as fp:
            people = pickle.load(fp)
    except:
        people = []
        with open('db/people.p', 'wb') as fp:
            pickle.dump(people, fp, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        with open('db/encodings.p', 'rb') as fp:
            encodings = pickle.load(fp)
    except:
        encodings = []
        with open('db/encodings.p', 'wb') as fp:
            pickle.dump(encodings, fp, protocol=pickle.HIGHEST_PROTOCOL)
    X = list()
    y = list()
    flag = list()
    # f = request.files['file']
    Files = request.files.getlist("files")
    N = len(Files)
    # f = Files[0]
    # f.save("hi.png")

    # f = request.files['files']
    ID = request.form.get("personID")
    ids = ID.split(',')
    # print(f.filename)
    # print(f)
    F = Files[0]
    imagelist = list()
    encode = list()
    if F.filename.rsplit('.', 1)[1].lower() in IMAGE_EXTENSIONS:
        for f in Files:
            if f.filename.rsplit('.', 1)[1].lower() in IMAGE_EXTENSIONS:

                f.save("static/added/"+"added_"+ID+".png")
                path = "static/added/" + "added_"+ID+".png"
                path2 = imageQuality(path)
                print(path2)
                path = path2
                face, images, results = extract_face(path)
                # encode = []
                if(len(face) >= 1):
                    # encode = ''
                    for i in range(len(images)):
                        print("i", i)
                        X = list()
                        y = list()
                        flag = list()
                        path = "static/extract/extracted_" + \
                            ids[i]+'_'+str(time.time())+".png"
                        # path2 = imageQuality(path)
                        # print(path2)
                        # path = path2
                        image = images[i].save(path)
                        imagelist.append(path)
                        X.extend(asarray(images[i]))
                        # print(type(X), type(face))
                        y.extend(ID)
                        X = asarray(X)
                        y = asarray(y)
                        # print('Loaded: ', X.shape, y.shape)
                        newTrainX = list()
                        print("x", X.shape)
                        # for face_pixels in X:
                        # print(face_pixels)
                        embedding = get_embedding(X)
                        newTrainX.append(embedding)
                        newTrainX = asarray(newTrainX)
                        # print(newTrainX)
                        encode.append(newTrainX)
                        # encode = encode+'\n' + \
                        #     str(ids[i]) + '\n' + str(newTrainX)
                        # print(newTrainX.shape)
                        encodings.append(newTrainX)
                        print(len(encodings))
                        # print(len(encodings))
                        people.append(ids[i])
                    with open('db/encodings.p', 'wb') as fp:
                        pickle.dump(encodings, fp,
                                    protocol=pickle.HIGHEST_PROTOCOL)
                    with open('db/people.p', 'wb') as fp:
                        pickle.dump(
                            people, fp, protocol=pickle.HIGHEST_PROTOCOL)
                        # print("hi")
                    h = "  encodings  added"

                else:
                    h = "encodings not added"
        Return = {"imagepath": imagelist, "encodings": encode, "heading": h}
        encodedNumpyData = json.dumps(Return, cls=NumpyArrayEncoder)
        return Response(encodedNumpyData)
        # return render_template("addface.html", results=imagelist, heading=h, encodings=encode)
    else:
        imagelist = list()
        encode = list()
        for f in Files:
            if f.filename.rsplit('.', 1)[1].lower() in VIDEO_EXTENSIONS:
                pathvideo = "static/addvideos/"+ID+str(time.time())+".mp4"
                f.save(pathvideo)
                vidcap = cv2.VideoCapture(pathvideo)
                success, image = vidcap.read()
                count = 0
                m = list()
                print(image.shape)
                while success:
                    X = list()
                    y = list()
                    success, image = vidcap.read()
                    try:
                        path = "static/images/"+ID+"_"+str(time.time())+".png"
                        cv2.imwrite(path, image)
                        path2 = imageQuality(path)
                        print(path2)
                        path = path2
                        face, images, results = extract_face(path)
                    except:
                        face = []

                    if len(face) >= 1:
                        # encode = ''
                        for i in range(len(images)):
                            print("i", i)
                            X = list()
                            y = list()
                            flag = list()
                            path = "static/extract/extracted_" + \
                                ids[i]+'_'+str(time.time())+".png"
                            # path2 = imageQuality(path)
                            # print(path2)
                            # path = path2
                            image = images[i].save(path)
                            imagelist.append(path)
                            X.extend(asarray(images[i]))
                            # print(type(X), type(face))
                            y.extend(ID)
                            X = asarray(X)
                            y = asarray(y)
                            # print('Loaded: ', X.shape, y.shape)
                            newTrainX = list()
                            print("x", X.shape)
                            # for face_pixels in X:
                            # print(face_pixels)
                            embedding = get_embedding(X)
                            newTrainX.append(embedding)
                            newTrainX = asarray(newTrainX)
                            # print(newTrainX)\
                            encode.append(newTrainX)
                            # encode = encode+'\n' + \
                            #     str(ids[i]) + '\n' + str(newTrainX)
                            # print(newTrainX.shape)
                            encodings.append(newTrainX)
                            print(len(encodings))
                            # print(len(encodings))
                            people.append(ids[i])
                        with open('db/encodings.p', 'wb') as fp:
                            pickle.dump(encodings, fp,
                                        protocol=pickle.HIGHEST_PROTOCOL)
                        with open('db/people.p', 'wb') as fp:
                            pickle.dump(
                                people, fp, protocol=pickle.HIGHEST_PROTOCOL)
                            # print("hi")
                        h = "encodings  added"

                    else:
                        m.append(0)
                        h = "encodings not added"
        Return = {"imagepath": imagelist, "encodings": encode, "heading": h}
        encodedNumpyData = json.dumps(Return, cls=NumpyArrayEncoder)
        return Response(encodedNumpyData)
        # return render_template("extractvideos.html", results=imagelist, videoname=pathvideo, heading=h)


@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    try:
        with open('db/people.p', 'rb') as fp:
            people = pickle.load(fp)
    except:
        people = []
        with open('db/people.p', 'wb') as fp:
            pickle.dump(people, fp, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        with open('db/encodings.p', 'rb') as fp:
            encodings = pickle.load(fp)
    except:
        encodings = []
        with open('db/encodings.p', 'wb') as fp:
            pickle.dump(encodings, fp, protocol=pickle.HIGHEST_PROTOCOL)
    Files = request.files.getlist("files")
    N = len(Files)
    imagelist = list()
    prob = []
    predicted = []
    F = Files[0]
    if F.filename.rsplit('.', 1)[1].lower() in IMAGE_EXTENSIONS:
        for f in Files:
            if f.filename.rsplit('.', 1)[1].lower() in IMAGE_EXTENSIONS:
                path = "static/detected/"+"detected"+str(time.time())+".png"
                f.save(path)
                path2 = imageQuality(path)
                print(path2)
                path = path2
                print(f.filename)
                X = list()
                face, images, results = extract_face(path)
                print(face, images)
                if len(face) >= 1:
                    for i in range(len(images)):
                        X = list()
                        print(face[i].shape)
                        path = "static/detected/"+"detected" + \
                            str(i)+"_"+str(time.time())+".png"

                        image = images[i].save(path)
                        imagelist.append(path)
                        X.extend(face[i])
                        X = asarray(X)
                        newTrainX = list()
                        # for face_pixels in X:
                        embedding = get_embedding(X)
                        newTrainX.append(embedding)
                        newTrainX = asarray(newTrainX)
                        in_encoder = Normalizer(norm='l2')
                        n = len(encodings)
                        print(type(encodings))
                        trainX = numpy.array(encodings)
                        trainX = numpy.resize(trainX, (n, 128))
                        # print(people)
                        trainy = numpy.array(people)
                        print(trainy.shape)
                        # trainX = numpy.resize(trainX, (1, 128))
                        print(type(trainX))
                        print(trainX.shape)
                        trainy = people
                        trainX = in_encoder.transform(trainX)
                        out_encoder = LabelEncoder()
                        out_encoder.fit(trainy)
                        model = SVC(kernel='linear', probability=True)
                        model.fit(trainX, trainy)
                        random_face_emb = newTrainX
                        print(newTrainX.shape)
                        samples = expand_dims(random_face_emb, axis=0)
                        print(samples.shape)
                        samples = numpy.resize(samples, (1, 128))
                        yhat_class = model.predict(samples)
                        yhat_prob = model.predict_proba(samples)
                        print(yhat_class, yhat_prob)
                        class_index = yhat_class[0]
                        print("class", class_index)
                        class_probability = max(yhat_prob[0]) * 100
                        print(class_probability)
                        predicted.append(yhat_class[0])
                        prob.append(class_probability)
                        print(prob)

                    h = "predicted : " + str(predicted) + \
                        "\nprobabilty : " + str(prob)
        Return = {"imagepath": imagelist, "heading": h}
        encodedNumpyData = json.dumps(Return, cls=NumpyArrayEncoder)
        return Response(encodedNumpyData)
        # return render_template("recognize.html", results=imagelist, heading=h)
    else:
        imagelist = list()
        for f in Files:
            if f.filename.rsplit('.', 1)[1].lower() in VIDEO_EXTENSIONS:
                pathvideo = "static/videos/"+f.filename+str(time.time())+".mp4"
                f.save(pathvideo)
                vidcap = cv2.VideoCapture(pathvideo)
                success, frame = vidcap.read()
                m = list()
                print(frame.shape)
                frame_array = list()
                while success:
                    X = list()
                    y = list()
                    success, frame = vidcap.read()
                    print("hi")

                    try:
                        path = "static/images/"+f.filename + \
                            "_"+str(time.time())+".png"
                        cv2.imwrite(path, frame)
                        path2 = imageQuality(path)
                        print(path2)
                        path = path2
                        face, images, results = extract_face(path)
                    except:
                        face = []
                    if(len(face) >= 1):
                        for i in range(len(images)):
                            X = list()
                            print(face[i].shape)
                            path = "static/detected/"+"detected" + \
                                str(i)+"_"+str(time.time())+".png"

                            image = images[i].save(path)
                            imagelist.append(path)
                            X.extend(face[i])
                            X = asarray(X)
                            newTrainX = list()
                            # for face_pixels in X:
                            embedding = get_embedding(X)
                            newTrainX.append(embedding)
                            newTrainX = asarray(newTrainX)
                            in_encoder = Normalizer(norm='l2')
                            n = len(encodings)
                            print(type(encodings))
                            trainX = numpy.array(encodings)
                            trainX = numpy.resize(trainX, (n, 128))
                            # print(people)
                            trainy = numpy.array(people)
                            print(trainy.shape)
                            # trainX = numpy.resize(trainX, (1, 128))
                            print(type(trainX))
                            print(trainX.shape)
                            trainy = people
                            trainX = in_encoder.transform(trainX)
                            out_encoder = LabelEncoder()
                            out_encoder.fit(trainy)
                            model = SVC(kernel='linear', probability=True)
                            model.fit(trainX, trainy)
                            random_face_emb = newTrainX
                            print(newTrainX.shape)
                            samples = expand_dims(random_face_emb, axis=0)
                            print(samples.shape)
                            samples = numpy.resize(samples, (1, 128))
                            yhat_class = model.predict(samples)
                            yhat_prob = model.predict_proba(samples)
                            print(yhat_class, yhat_prob)
                            class_index = yhat_class[0]
                            print("class", class_index)
                            class_probability = max(yhat_prob[0]) * 100
                            print(class_probability)
                            predicted.append(yhat_class[0])
                            prob.append(class_probability)
                            print(prob)
                            h = "predicted : " + str(predicted) + \
                                "\nprobabilty : " + str(prob)
                            file = draw_image_with_boxes(frame, results, h)
                            file = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)
                        frame_array.append(file)
                out_video = numpy.empty(
                    [len(frame_array), height1, width1, 3], dtype=numpy.uint8)
                out_video = out_video.astype(numpy.uint8)
                for i in range(len(frame_array)):
                    out_video[i] = frame_array[i]

                skvideo.io.vwrite(pathvideo, out_video)
        Return = {"imagepath": imagelist, "heading": h}
        encodedNumpyData = json.dumps(Return, cls=NumpyArrayEncoder)
        return Response(encodedNumpyData)
        # return render_template("extractvideos.html", results=imagelist, videoname=pathvideo, heading=h)


@ app.route('/removeencodings', methods=['GET', 'POST'])
def removeencodings():
    try:
        with open('db/people.p', 'rb') as fp:
            people = pickle.load(fp)
    except:
        people = []
        with open('db/people.p', 'wb') as fp:
            pickle.dump(people, fp, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        with open('db/encodings.p', 'rb') as fp:
            encodings = pickle.load(fp)
    except:
        encodings = []
        with open('db/encodings.p', 'wb') as fp:
            pickle.dump(encodings, fp, protocol=pickle.HIGHEST_PROTOCOL)
    X = list()
    # f = request.files['file']
    Files = request.files.getlist("files")
    # print(f.filename)
    Total = 0
    for f in Files:
        f.save("static/images/"+f.filename)
        x = "static/images/"+f.filename
        path2 = imageQuality(x)
        x = path2
        face, images, results = extract_face(x)
        if len(face) < 1:
            print("No faces found.")
            os.remove(os.path.join("static/images/", f.filename))
            # return "No faces found."
        else:
            for i in range(len(images)):
                X = list()
                encoding_num = 0
                encode = []
                # if(len(face) == 1):
                X.extend(face[i])
                X = asarray(X)
                print(type(X))
                print('Loaded: ', X.shape)
                newTrainX = list()
                # for face_pixels in X:
                embedding = get_embedding(X)
                newTrainX.append(embedding)
                image_encoding = newTrainX[0]
                i = 0
                print(embedding)
                # print(image_encoding)
                for _ in range(len(encodings)):
                    print(encodings[i])
                    # print(encodings[i][0])
                    if str(encodings[i][0]) == str(image_encoding):
                        del encodings[i]
                        del people[i]
                        i -= 1
                        encoding_num += 1
                        Total = Total+1
                    i += 1
            os.remove(os.path.join("static/images/", f.filename))
    with open('db/encodings.p', 'wb') as fp:
        pickle.dump(encodings, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open('db/people.p', 'wb') as fp:
        pickle.dump(people, fp, protocol=pickle.HIGHEST_PROTOCOL)

    return jsonify("Removed "+str(encoding_num)+" image encodings.")


def photo():
    return render_template("webcam_page.html")


@ app.route('/webcam', methods=['GET', 'POST'])
def webcam():
    try:
        with open('db/people.p', 'rb') as fp:
            people = pickle.load(fp)
    except:
        people = []
        with open('db/people.p', 'wb') as fp:
            pickle.dump(people, fp, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        with open('db/encodings.p', 'rb') as fp:
            encodings = pickle.load(fp)
    except:
        encodings = []
        with open('db/encodings.p', 'wb') as fp:
            pickle.dump(encodings, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print("[INFO] starting video stream thread...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    # loop over frames from the video stream
    i = 0
    imagelist = list()
    frames = list()
    frame_array = list()
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # show the frame
        cv2.imshow("Frame", frame)
        path = "static/png/"+str(i)+str(time.time())+".png"
        cv2.imwrite(path, frame)
        imagelist.append(path)
        frames.append(frame)
        key = cv2.waitKey(1) & 0xFF
        i = i+1
        time.sleep(0.5)

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    res = list()
    pathvideo = "static/png/videos/"+str(time.time())+".mp4"
    for i in range(len(imagelist)):
        X = list()
        face, images, results = extract_face(imagelist[i])
        if(len(face) == 1):
            X.extend(face)
            X = asarray(X)
            newTrainX = list()
            for face_pixels in X:
                embedding = get_embedding(face_pixels)
                newTrainX.append(embedding)
            newTrainX = asarray(newTrainX)
            in_encoder = Normalizer(norm='l2')
            n = len(encodings)
            trainX = numpy.array(encodings)
            trainX = numpy.resize(trainX, (n, 128))
            trainy = numpy.array(people)
            trainy = people
            trainX = in_encoder.transform(trainX)
            out_encoder = LabelEncoder()
            out_encoder.fit(trainy)
            model = SVC(kernel='linear', probability=True)
            model.fit(trainX, trainy)
            random_face_emb = newTrainX
            print(newTrainX.shape)
            samples = expand_dims(random_face_emb, axis=0)
            print(samples.shape)
            samples = numpy.resize(samples, (1, 128))
            yhat_class = model.predict(samples)
            yhat_prob = model.predict_proba(samples)
            print(yhat_class, yhat_prob)
            class_index = yhat_class[0]
            print("class", class_index)
            class_probability = max(yhat_prob[0]) * 100
            print(class_probability)
            # predict_names = out_encoder.inverse_transform(yhat_class)
            print('Predicted: %s (%.3f)' % (yhat_class[0], class_probability))
            h = "predicted : " + \
                str(yhat_class[0]) + "\nprobabilty : " + str(class_probability)
            res.append(h)
            file = draw_image_with_boxes(frames[i], results, h)
            # cv2.imshow("1",file)
            # cv2.waitKey(1)
            file = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)
            frame_array.append(file)
    out_video = numpy.empty(
        [len(frame_array), height1, width1, 3], dtype=numpy.uint8)
    out_video = out_video.astype(numpy.uint8)
    for i in range(len(frame_array)):
        out_video[i] = frame_array[i]
    print(pathvideo)
    skvideo.io.vwrite(pathvideo, out_video)
    Return = {"imagepath": imagelist, "heading": res}
    encodedNumpyData = json.dumps(Return, cls=NumpyArrayEncoder)
    return Response(encodedNumpyData)
    # return render_template("extractvideos.html", results=imagelist, videoname=pathvideo, heading=h)


@app.route('/removePersonID', methods=['GET', 'POST'])
def removepersonID():
    try:
        with open('db/people.p', 'rb') as fp:
            people = pickle.load(fp)
    except:
        people = []
        with open('db/people.p', 'wb') as fp:
            pickle.dump(people, fp, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        with open('db/encodings.p', 'rb') as fp:
            encodings = pickle.load(fp)
    except:
        encodings = []
        with open('db/encodings.p', 'wb') as fp:
            pickle.dump(encodings, fp, protocol=pickle.HIGHEST_PROTOCOL)
    ID = request.form.get("personID")
    flag = False
    if ID in people:
        flag = True
        i = 0
        for _ in range(len(people)):
            if people[i] == ID:
                del encodings[i]
                i -= 1
            i += 1
    del people[people.index(ID)]

    with open('db/encodings.p', 'wb') as fp:
        pickle.dump(encodings, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open('db/people.p', 'wb') as fp:
        pickle.dump(people, fp, protocol=pickle.HIGHEST_PROTOCOL)

    return jsonify({ID: ID})
###################################################################


@app.route('/getAllFaceEncodings', methods=['GET', 'POST'])
def getallencodings():
    try:
        with open('db/people.p', 'rb') as fp:
            people = pickle.load(fp)
    except:
        people = []
        with open('db/people.p', 'wb') as fp:
            pickle.dump(people, fp, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        with open('db/encodings.p', 'rb') as fp:
            encodings = pickle.load(fp)
    except:
        encodings = []
        with open('db/encodings.p', 'wb') as fp:
            pickle.dump(encodings, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print(len(encodings), len(people))

    ID = request.form.get("personID")
    encodingslist = list()
    faceslist = list()

    flag = False
    if ID in people:
        flag = True
        for i in range(len(people)):
            if people[i] == ID:
                encodingslist.append(encodings[i])
                path = "static/added/added_"+ID+".png"
                image = Image.open(path)
                image = image.convert('RGB')
                faceslist.append(image)

    Return = {"encodingslist": encodingslist}
    encodedNumpyData = json.dumps(Return, cls=NumpyArrayEncoder)
    return Response(encodedNumpyData)


if __name__ == "__main__":
    app.run(port=4001, debug=True)

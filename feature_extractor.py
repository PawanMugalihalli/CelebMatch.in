import os
import pickle
import face_recognition
from tqdm import tqdm
import cv2

actors = os.listdir('data')

filenames = []

for actor in actors:
    filenames.append(os.path.join('data',actor))

pickle.dump(filenames,open('filenames.pkl','wb'))

filenames = pickle.load(open('filenames.pkl','rb'))

def feature_extractor(img_path):
    img=cv2.imread(img_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    feature=[]
    if len(face_recognition.face_encodings(img)):
        feature=face_recognition.face_encodings(img)[0]
    return feature

features = []
for file in tqdm(filenames):
    features.append(feature_extractor(file))

pickle.dump(features,open('embedding.pkl','wb'))


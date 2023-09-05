from flask import Flask
import ml
import api

app = Flask(__name__)

rainRecord = [18.5, 0.7, 0, 0, 0, 0, 0, 0, 103, 19.1]
rainRecordIdx = 0


predRecord = [1.5, 1.3, 1.9,  2.3,  3.4,  5.4,  1.2, 1.3, 5.6,   8.1,  9.3,  10.2, 11.4,  9.5,  8.7,  7.3, 6.5, 4.3, 2.1, 1.3]

@app.route("/rain", methods=['GET'])
def getRain():
  #########
  global rainRecordIdx
  res = api.getRainFromDummy(rainRecordIdx)
  rainRecordIdx += 1
  #########
  # res = api.getRainFromApi()
  rainRecord.pop(0)
  rainRecord.append(res)
  
  
  # prediction = ml.fl.predict(rainRecord)
  prediction = [predRecord[rainRecordIdx%20]]
  return str(res) + "," + str(prediction[0])



@app.route("/waterlevel", methods=['GET'])
def getWaterlevel():
  prediction = ml.fl.predict(rainRecord)
  print(rainRecord)
  return str(prediction[0])




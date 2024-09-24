from roboflow import Roboflow
rf = Roboflow(api_key="OlFtojxybMFMxBTfdLEF")
project = rf.workspace("education-oqzdo").project("snack-eej94")
version = project.version(1)
dataset = version.download("yolov9") 
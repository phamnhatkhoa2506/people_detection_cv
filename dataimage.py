import os
from roboflow import Roboflow
from dotenv import load_dotenv


if __name__ == '__main__':
    load_dotenv()

    rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))
    project = rf.workspace("leo-ueno").project("people-detection-o4rdr")
    version = project.version(1)
    dataset = version.download("yolov11")
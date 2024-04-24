import requests
from ultralytics import YOLO
import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json 

    
model = YOLO("weights-ds2.pt")


app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProcessRequest(BaseModel):
    url: str


@app.get("/")
def check():
    return {"message": "Hello World!"}

def get_prediction():
    results = model('image.jpg')
    print(results[0].tojson())
    return results[0].tojson()

@app.post("/process_picture/")
async def get_img_direct(image: UploadFile = File(...)):
    try:
        with open('image.jpg', 'wb') as handler:
            handler.write(image.file.read())
        result = json.loads(get_prediction())
        classnames = {result[i]["name"] : result[i]["confidence"]for i in range(len(result))}
        return {"response": classnames}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
   uvicorn.run("main:app", port=8000, log_level="info", reload=True)

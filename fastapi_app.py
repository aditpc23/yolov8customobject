import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

# Directory to save uploaded images
UPLOAD_DIR = "uploaded_images"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

app = FastAPI()

@app.post("/upload")
async def upload_image(imageFile: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, imageFile.filename)
    with open(file_location, "wb") as f:
        f.write(await imageFile.read())
    return JSONResponse(content={"info": f"file '{imageFile.filename}' saved at '{file_location}'"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

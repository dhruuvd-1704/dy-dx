from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile
import os

app = FastAPI()

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(await file.read())
        video_path = temp_file.name
        
        # Print the path to check if file is saved correctly
        print(f"Video saved at: {video_path}")

        # List files in the temporary directory
        temp_dir = os.path.dirname(video_path)
        print(f"Files in temporary directory ({temp_dir}):")
        for file in os.listdir(temp_dir):
            print(file)

        # Placeholder for detection logic
        detection_result = "Placeholder result"  # Replace with actual detection logic
        
        # Return the path for debugging purposes
        return JSONResponse(content={"result": detection_result, "video_path": video_path})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    
    
    
    
    





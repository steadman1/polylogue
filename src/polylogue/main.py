from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    print("Hello from polylogue!")
    return {"message": "Hello World"}

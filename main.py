from fastapi import FastAPI, Request
from src.apis import web_api
import uvicorn

app = FastAPI()

app.include_router(web_api.router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)
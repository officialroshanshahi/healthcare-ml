# uvicorn_init.py
import uvicorn
from src.v1.config import HOST, PORT

if __name__ == "__main__":
    uvicorn.run(
        "api.api:app",
        host=HOST,
        port=PORT,
        reload=True,
        reload_dirs=["src", "api"],
        reload_excludes=[
            "**/active_model.json", 
            "**/models/**", 
            "**/plots/**",
            "**/data/**"
        ]
    )
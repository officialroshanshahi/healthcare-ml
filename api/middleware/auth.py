import time
import jwt
from collections import defaultdict, deque
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from src.v1.config import JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRY_SECONDS, LIMIT

# taking out the bearer token
security_schema = HTTPBearer(auto_error=False)
hits = defaultdict(deque)


# providing access token to users
def create_access_token(username: str) -> str:    
    payload = {
        "sub": username,
        "exp": time.time() + JWT_EXPIRY_SECONDS
    }
    
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


# auth logic
def auth(credentials: HTTPAuthorizationCredentials = Depends(security_schema)):
    
    # if no credentials are provided they are unauthorized
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Please input your Bearer JWT token."
        )
    
    # taking token out
    token = credentials.credentials
    
    # checking if username is present
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username = payload.get("sub")
        
        if not username:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload.")
    
    #handling expired token to change the token time limit change .env file
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired.")
    
    # other errors handling
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token.")

    # rate limiting logic
    now = time.time()
    
    q = hits[username]
    
    while q and now - q[0] > 60:
        q.popleft()
    
    if len(q) >= LIMIT: # limit is currently 60 times a minute
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded.")
    
    q.append(now)

    return username
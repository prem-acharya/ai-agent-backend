from fastapi import APIRouter, HTTPException, Header, Query, Depends, Response
from fastapi.responses import RedirectResponse
from typing import Dict, Optional
from pydantic import BaseModel

from src.auth.google_oauth import GoogleOAuth

router = APIRouter()
oauth_handler = GoogleOAuth()

class TokenRequest(BaseModel):
    code: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int

class RefreshRequest(BaseModel):
    refresh_token: str

@router.get("/auth-url")
async def get_auth_url():
    """
    Get the URL for Google OAuth authorization
    """
    try:
        auth_url = oauth_handler.get_authorization_url()
        return {"auth_url": auth_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get auth URL: {str(e)}")

@router.post("/token", response_model=TokenResponse)
async def exchange_token(request: TokenRequest):
    """
    Exchange the authorization code for tokens
    """
    try:
        tokens = oauth_handler.exchange_code_for_tokens(request.code)
        return {
            "access_token": tokens.get("access_token"),
            "refresh_token": tokens.get("refresh_token"),
            "token_type": tokens.get("token_type"),
            "expires_in": tokens.get("expires_in")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to exchange token: {str(e)}")

@router.post("/refresh")
async def refresh_token(request: RefreshRequest):
    """
    Refresh the access token using a refresh token
    """
    try:
        tokens = oauth_handler.refresh_access_token(request.refresh_token)
        return {
            "access_token": tokens.get("access_token"),
            "token_type": tokens.get("token_type"),
            "expires_in": tokens.get("expires_in")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh token: {str(e)}")

@router.post("/revoke")
async def revoke_token(token: str):
    """
    Revoke the specified token
    """
    try:
        success = oauth_handler.revoke_token(token)
        if success:
            return {"message": "Token revoked successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to revoke token")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to revoke token: {str(e)}")

@router.get("/callback")
async def auth_callback(code: str = Query(...), state: Optional[str] = Query(None)):
    """
    Handle the OAuth callback from Google
    This endpoint exists for demonstration purposes - typically, 
    the OAuth callback is handled by the frontend
    """
    try:
        # In a real-world scenario, you might want to store the tokens
        # or send them to the frontend via a secure method
        tokens = oauth_handler.exchange_code_for_tokens(code)
        return {"message": "Authentication successful", "access_token": tokens.get("access_token")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Callback error: {str(e)}")

import os
import requests
from typing import Dict, Any, Optional
from fastapi import HTTPException

class GoogleOAuth:
    """
    Handles OAuth authentication with Google API
    """
    
    def __init__(self):
        self.client_id = os.getenv("GOOGLE_CLIENT_ID")
        self.client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        self.redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")
        self.oauth_scopes = os.getenv("GOOGLE_OAUTH_SCOPES")
        
        if not all([self.client_id, self.client_secret, self.redirect_uri, self.oauth_scopes]):
            raise ValueError("Google OAuth credentials not properly configured")
    
    def get_authorization_url(self) -> str:
        """
        Returns the URL for Google OAuth authorization
        """
        auth_url = "https://accounts.google.com/o/oauth2/auth"
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": self.oauth_scopes,
            "response_type": "code",
            "access_type": "offline",
            "prompt": "consent"
        }
        
        # Build the URL
        query_string = "&".join([f"{key}={value}" for key, value in params.items()])
        return f"{auth_url}?{query_string}"
    
    def exchange_code_for_tokens(self, code: str) -> Dict[str, Any]:
        """
        Exchange the authorization code for access and refresh tokens
        """
        token_url = "https://oauth2.googleapis.com/token"
        data = {
            "code": code,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": self.redirect_uri,
            "grant_type": "authorization_code"
        }
        
        response = requests.post(token_url, data=data)
        
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to exchange code for tokens: {response.text}")
        
        return response.json()
    
    def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """
        Use the refresh token to get a new access token
        """
        token_url = "https://oauth2.googleapis.com/token"
        data = {
            "refresh_token": refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "refresh_token"
        }
        
        response = requests.post(token_url, data=data)
        
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to refresh token: {response.text}")
        
        return response.json()
    
    def revoke_token(self, token: str) -> bool:
        """
        Revoke the specified token
        """
        revoke_url = "https://oauth2.googleapis.com/revoke"
        params = {"token": token}
        
        response = requests.post(revoke_url, params=params)
        
        return response.status_code == 200

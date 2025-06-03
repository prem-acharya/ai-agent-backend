# Backend Setup Info

## AI Agent Backend

This backend powers the AI Agent that integrates with **Google Meet**, **Google Tasks**, and other tools to execute using LLMs.

It's a **FastAPI** service that connects:
- LLMs (GPT-4o, Gemini 2.0 Flash)
- Google APIs (Calendar, Tasks)
- External tools (weather API, time services)
- LangChain / LangGraph frameworks

---

## Features

- Handle LLM prompt input with system prompting
- Schedule meetings and reminders via Google APIs
- Stream LLM responses
- Get contextual data (time/weather/search)
- Supports GPT-4o and Gemini 2.0 Flash

---

## Tech Stack

- **API Framework**: FastAPI (Python)
- **LLM**: OpenAI GPT-4o, Gemini 2.0 Flash
- **Orchestration Framework**: LangChain, LangGraph
- **Auth**: Google OAuth tokens from frontend (via Clerk)

---

## Google API Setup

Make sure you:
- Register your app in Google Cloud Console
- Enable the following APIs:
  - Google Calendar API
  - Google Tasks API
- Store credentials securely (e.g. `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`)
- Use OAuth tokens passed from the frontend after Clerk login

---

## 📬 Contact

For backend logic, API issues, or LLM pipeline help:

- GitHub Issues
- PRs welcome for new agents or tool integrations

# Friend AI Chatbot

An AI chatbot that learns to mimic your friend's communication style based on conversation data you provide.

## Features

- Upload conversation data (chats, emails, social media posts)
- AI learns your friend's communication style
- Chat with the AI that responds like your friend would
- Secure data handling with MongoDB

## Technology Stack

- **Backend**: FastAPI, Python
- **AI Models**: GPT-2, Sentence Transformers (all-MiniLM-L6-v2)
- **Database**: MongoDB Atlas
- **Deployment**: Railway

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables:
   - `MONGODB_URI`: Your MongoDB connection string
4. Run the application: `uvicorn main:app --reload`

## Deployment

This project is configured for deployment on Railway.

## License

MIT

import logging

from fastapi import FastAPI

from apis import chat, documents

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO)

def configure_application():
    application = FastAPI(
        title='Financial ai assistant',
        debug=False,
    )

    # Routers
    application.include_router(chat.router)
    application.include_router(documents.router)

    # Logging
    logger = logging.getLogger("uvicorn.error")
    logger.propagate = False

    return application

app = configure_application()

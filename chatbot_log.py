import logging

logging.basicConfig(filename='chatbot.log', level=logging.DEBUG)
logging.debug(f"Loss at step {step}: {loss}")
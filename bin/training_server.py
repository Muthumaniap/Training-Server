import logging.config
import os
import javaproperties

from flask import Flask
from flask_socketio import SocketIO
from global_variables import Globals
from classifier_model import classifier_model

# define the conf, data and log paths
Globals.conf = os.path.dirname(os.path.realpath(__file__)).replace("bin", "conf")
Globals.data = os.path.dirname(os.path.realpath(__file__)).replace("bin", "data")
Globals.log = os.path.dirname(os.path.realpath(__file__)).replace("bin", "logs")

# flask application
app = Flask(__name__)

# socket application
Globals.socket_conn = socketapp = SocketIO(app)

@app.get("/train")
def model_status():
    classifier_model()
    return {"Classifier": "Trained"}

@app.get("/get-test/{inputData}")
def model_test_status():
    return {"Classifier": "Tested"}

if __name__ == "__main__":

    try:
        # disable the info logs of sockets and web server
        logging.basicConfig(level=logging.WARNING)
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.WARNING)

        # Load the logging configuration from the file
        logging.config.fileConfig(os.path.join(Globals.conf, "logging.conf"))
        logging = logging.getLogger()

        logging.info("Entering the main function")

        with open(
            os.path.join(Globals.conf, "training_server.properties"), encoding="utf-8"
        ) as prop_file:
            properties = javaproperties.load(prop_file)
        logging.debug(properties)

        logging.info(
            "Starting server at %s:%s",
            properties["training.server.ip"],
            properties["training.server.port"],
        )
        socketapp.run(
            app,
            allow_unsafe_werkzeug=True,
            host=properties["training.server.ip"],
            port=int(properties["training.server.port"]),
        )

        app.run()

    except Exception as error:
        logging.exception("Could not start the server: %s", error)

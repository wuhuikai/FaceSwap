import base64
import glob
import json
import logging
import os
import tempfile

import cv2

from flask import (
    Flask,
    jsonify,
    abort,
    make_response,
    request,
    send_file,
)
from fuzzywuzzy import process

from face.face_detection import select_face, select_face_update
from face.face_swap import face_swap
from utils.helpers import download_with_user_agent
from random import random


NOT_FOUND = "Not found"
BAD_REQUEST = "Bad request"
HIT_PROBABILITY = .4

STATISTICS_TABLE= {}

app = Flask(__name__)

####################################################
# Load constants once
####################################################
if os.getenv("ENVIRONMENT", "") == "container":
    people_path = "/people/*.jpg"
    lookup_file = "/models/lookup.json"
else:
    people_path = "../people/*.jpg"
    lookup_file = "../models/lookup.json"
PEOPLE = [os.path.basename(person) for person in glob.glob(people_path)]
logging.basicConfig(level=logging.INFO)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({"error": NOT_FOUND}), 404)


@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({"error": BAD_REQUEST}), 400)


@app.route("/status")
def get_health():
    return "Health Check OK", 200


@app.route("/image/<id>", methods=["GET"])
def image(id):
    id_decoded = base64.b64decode(bytes(id, "utf-8")).decode("utf-8")
    print(id_decoded)
    if os.path.isfile(id_decoded):
        return send_file(id_decoded, mimetype="image/jpeg", attachment_filename="test.jpg")
    else:
        return not_found()


@app.route("/snowball", methods=["POST"])
def snowball():
    probability = random()
    logging.info(request.form)
    request_text = request.form["text"]
    request_text = request_text.replace("\xa0", " ").replace("<", " ").replace(">", " ")
    request_text = " ".join(request_text.split())
    
    current_user = clean_name(request.form['user_name'])

    if not STATISTICS_TABLE.get(current_user):
        STATISTICS_TABLE[current_user] = {'Hit': 0, 'Attempt': 0}
 
    if 'stats' == request_text:
        return render_stats(current_user)

    if 'rankings' == request_text:
        return render_rankings()

    target_name = clean_name(request_text)
    if target_name == current_user:
        message = f"Why are you trying to hit yourself silly? Throw a snowball at someone else!"
        return render_message(message)

    message = outcomes(probability, current_user, target_name)
    return render_message(message)


def clean_name(potential_name):
    if "|" in potential_name:
        name = potential_name.split("|")[1].replace(">", "").replace(".", " ").title()
    else:
        name = potential_name.replace(">", "").replace(".", " ").title()
    return name

def render_rankings():
    message = "You must throw at least once to be ranked.\n"
    filtered_STATISTICS_TABLE = {k:v for k,v in STATISTICS_TABLE.items() if v['Attempt']!=0}
    rankings_table = ''.join([ f"{key} Successful Hit: {value['Hit']} Attempts: {value['Attempt']}\n" for key, value in sorted(filtered_STATISTICS_TABLE.items(), key=lambda item: item[1]['Hit'])][:10])

    return render_message(message + rankings_table)

def render_stats(current_user):
    if STATISTICS_TABLE[current_user]['Attempt'] == 0:
        return render_message("ERROR, DATA NOT FOUND \nAre we human? Or are we dancer?")

    accuracy = STATISTICS_TABLE[current_user]['Hit']/ STATISTICS_TABLE[current_user]['Attempt']

    if accuracy > .9:
        message = f'Turn off your hacks or you will get nerfed!'
    elif accuracy > .5:
        message = f'You have a great throw! Keep it up!'
    elif accuracy > .3:
        message = f'You are alright. Statistically speaking you are very much just only alright.'
    elif accuracy > .1:
        message = f'Probably work on your aim during your freetime.'
    else:
        if STATISTICS_TABLE[current_user]['Attempt'] > 10:
            message = f'Hey buddy, everything alright? Consider bribing someone... You probably need help.'
        else: 
            message = f'Keep trying! May the odds always be in your favor!'

    stat_table = '\n'+ ''.join([ f'{key}: {value}    '  for key, value in STATISTICS_TABLE[current_user].items()]) + f'\n {current_user} has an accuracy of {accuracy:.2f}'
    return render_message(message + stat_table)


def render_message(message):
    return jsonify(
                      {
                          "response_type": "in_channel",
                          "text": f"{message}",
                      }
                  )

    
def outcomes(probability, current_user, target):
    STATISTICS_TABLE[current_user]['Attempt'] += 1
    if probability < HIT_PROBABILITY:
        message = f"You hit {target} square in the back of the head. {target} is secretly crying right now."
        STATISTICS_TABLE[current_user]['Hit'] += 1
    else:
        if probability < .6:
            message = f"You tripped and failed to hit your target, {target} is laughing at you from afar."
        elif probability < .8:
            if target != "Stanley Phu" and current_user:
                person_hit = "Stanley Phu"
            else:
                person_hit = "Yen-Ting Chen"
            message = f"You hit the ceiling, it bounces, and hits {person_hit} on the face instead. Try again maybe?"
        elif probability < .9:
            message = f"You tried to hit {target} but hit the monitor instead. You may or may not have left a dent on that monitor."
        else:
            message = f'As Simon would say, "learn to aim dude". So toxic. I apologize in his stead. You missed.'
    return message


@app.route("/swap", methods=["POST"])
def swap():

    if not request.form:
        abort(400)
    if not request.form["text"]:
        abort(400)

    logging.info(request.form)
    request_text = request.form["text"]
    request_text = request_text.replace("\xa0", " ").replace("<", " ").replace(">", " ")
    request_text = " ".join(request_text.split())
    dst_name_or_url = request_text.split(" ")[0]
    src_name_or_url = request_text.split(" ")[1]

    warp_2d = False
    correct_color = False

    if warp_2d in request_text.split(" "):
        warp_2d = True

    if correct_color in request_text.split(" "):
        correct_color = True

    logging.info("Request: " + request_text)
    logging.info(dst_name_or_url)
    logging.info(src_name_or_url)


    # Need to use a helper to download the images to fake a browser (some websites block straight downloads)
    with tempfile.NamedTemporaryFile(suffix=".jpg") as dest_img_file:
        with tempfile.NamedTemporaryFile(suffix=".jpg") as src_img_file:
            if dst_name_or_url.lower().startswith("http"):
                download_with_user_agent(dst_name_or_url, dest_img_file)
                dst_img = cv2.imread(dest_img_file.name)
            else:
                dst_img = cv2.imread("../people/" + _find_person(dst_name_or_url))

            if src_name_or_url.lower().startswith("http"):
                download_with_user_agent(src_name_or_url, src_img_file)
                src_img = cv2.imread(src_img_file.name)
            else:
                src_img = cv2.imread("../people/" + _find_person(src_name_or_url))

            src_points, src_shape, src_face = select_face(src_img)  # Select src face
            dest_faces = select_face_update(dst_img)  # Select dst face

            if src_points is None:
                logging.info("Detect 0 Face !!!")
                abort(400)

            for face in dest_faces:
                dst_points, dst_shape, dst_face = face
                dst_img = face_swap(src_face, dst_face, src_points, dst_points, dst_shape, dst_img, warp_2d, correct_color)

            tmp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            cv2.imwrite(tmp_file.name, dst_img)

            tmp_file_encoded = base64.b64encode(tmp_file.name.encode("utf-8")).decode("utf-8")

            json_return = jsonify(
                {
                    "response_type": "in_channel",
                    "attachments": [{"image_url": f"https://gary-robot.herokuapp.com/image/{tmp_file_encoded}"}],
                }
            )
            logging.info(tmp_file_encoded)

            return (
                json_return,
                200,
            )
    return make_response(jsonify({"error": BAD_REQUEST}), 400)


def _find_person(name):
    if "|" in name:
        file_name = name.split("|")[1].replace(">", "").replace(".", " ").title() + ".jpg"
    else:
        file_name = name.replace(">", "").replace(".", " ").title() + ".jpg"

    logging.info(file_name)
    if os.path.isfile("../people/" + file_name):
        best_match = file_name
    else:  # Fuzzy match names
        best_match, _ = process.extractOne(name, PEOPLE)
    return best_match


if __name__ == "__main__":
    app.run(debug=True)

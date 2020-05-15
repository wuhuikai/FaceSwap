import base64
import glob
import logging
import os
import tempfile
from datetime import datetime
from random import random

import cv2
import requests
from flask import Flask, abort, jsonify, make_response, request, send_file
from fuzzywuzzy import process

from face.face_detection import select_face, select_face_update
from face.face_swap import face_swap
from utils.helpers import download_with_user_agent

NOT_FOUND = "Not found"
BAD_REQUEST = "Bad request"
FORBIDDEN = "Forbidden"
HIT_PROBABILITY = 0.6
FRISBEE_HOLDER = {}
TOSS_PROBABILITY = 0.8
FRISBEE_TOKEN = os.environ.get("FRISBEE_TOKEN")

SNOWBALL_TABLE = {}

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
def not_found():
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


@app.route("/frisbee", methods=["POST"])
def throw():
    logging.info(request.form)
    request_text = request.form["text"]

    if not request.form["token"] == "hv1Ga1tpwrqIK7np4LxiLu45":
        return make_response(jsonify({"error": FORBIDDEN}), 403)

    probability = random()

    current_channel_id = request.form["channel_id"]

    current_user = clean_name(request.form["user_name"])
    target_name = clean_name(request_text)
    target_user_id = get_user_id(request_text)

    if not FRISBEE_HOLDER.get(current_channel_id):
        members = fetch_member(current_channel_id)
        FRISBEE_HOLDER[current_channel_id] = {"frisbee_holder": None, "chain": 0, "members": members}

    if request_text == "refresh" and (current_user == "Stanley Phu" or current_user == "Kevin Hsieh"):
        members = fetch_member(current_channel_id)
        FRISBEE_HOLDER[current_channel_id] = {"frisbee_holder": None, "chain": 0, "members": members}
        message = "Refreshing..... Refreshed"
        return render_message(message)

    if request_text == "reset" and (current_user == "Stanley Phu" or current_user == "Kevin Hsieh"):
        members = FRISBEE_HOLDER[current_channel_id]["members"]
        FRISBEE_HOLDER[current_channel_id] = {"frisbee_holder": None, "chain": 0, "members": members}
        message = "The game has been reset."
        return render_message(message)

    if len(request_text.split("|")) < 2 or not target_user_id:
        return render_message("You have to toss this to someone with the @ handle!")

    if current_user == target_name:
        return render_message("Quit hogging the disc asshole!")

    logging.info(FRISBEE_HOLDER)
    logging.info(current_user)
    logging.info(target_name)

    if (not FRISBEE_HOLDER[current_channel_id]["frisbee_holder"]) or (
        FRISBEE_HOLDER[current_channel_id]["frisbee_holder"] == current_user
    ):
        if target_user_id in FRISBEE_HOLDER[current_channel_id]["members"]:
            success, message = frisbee_outcomes(probability, target_name)
            if success:
                FRISBEE_HOLDER[current_channel_id]["frisbee_holder"] = target_name
                FRISBEE_HOLDER[current_channel_id]["chain"] += 1
                if FRISBEE_HOLDER[current_channel_id]["chain"] > 2:
                    FRISBEE_HOLDER[current_channel_id]["chain"] = 0
                    FRISBEE_HOLDER[current_channel_id]["frisbee_holder"] = None
                    message += f"\n   :tada: YOU GUYS SCORED! :tada:"
            else:
                FRISBEE_HOLDER[current_channel_id]["chain"] = 0
                FRISBEE_HOLDER[current_channel_id]["frisbee_holder"] = None
                message += f"\nSomeone pick up the disc and throw it!"
        else:
            message = "Toss it to someone in the channel, loser."
    else:
        message = "You can't toss what you don't have, sucker!"

    return render_message(message)


def fetch_member(current_channel_id):
    try:
        response = requests.get(
            "https://slack.com/api/conversations.members", {"token": FRISBEE_TOKEN, "channel": current_channel_id}
        )
        return response.json()["members"]
    except Exception as e:
        logging.info(e)
        return render_message("Can't use in DMs")


@app.route("/snowball", methods=["POST"])
def snowball():
    probability = random()
    logging.info(request.form)
    logging.info(probability)
    request_text = request.form["text"]

    if not request.form["token"] == "40gIKRHZUlj1e5r9Ya4m5X9Z":
        return make_response(jsonify({"error": FORBIDDEN}), 403)

    request_text = request_text.replace("\xa0", " ").replace("<", " ").replace(">", " ")
    request_text = " ".join(request_text.split())

    current_user = clean_name(request.form["user_name"])
    target_name = clean_name(request_text)

    if target_name == current_user:
        message = f"Why are you trying to hit yourself silly? Throw a snowball at someone else!"
        return render_message(message)

    if not SNOWBALL_TABLE.get(current_user):
        SNOWBALL_TABLE[current_user] = {"Hit": 0, "Attempt": 0, "Combo": 0}

    if "stats" == request_text:
        return render_stats(current_user)

    if "rankings" == request_text:
        return render_rankings()

    if not SNOWBALL_TABLE.get(target_name):
        SNOWBALL_TABLE[target_name] = {"Hit": 0, "Attempt": 0, "Combo": 0}

    if SNOWBALL_TABLE[current_user].get("Stunned_Time"):
        time_diff_in_seconds = (datetime.now() - SNOWBALL_TABLE[current_user]["Stunned_Time"]).seconds
        if time_diff_in_seconds <= 300:
            message = (
                f"You are stunned because some guy threw a golden snowball at you and knocked you out. What a dick!"
            )
            message += f"\n"
            message += f"You can't do anything for the next {(300 - time_diff_in_seconds)} seconds"
            return render_message(message)

    message = snowball_outcomes(probability, current_user, target_name)
    return render_message(message)


def clean_name(potential_name):
    if "|" in potential_name:
        name = potential_name.split("|")[1].replace(">", "").replace(".", " ").title()
    else:
        name = potential_name.replace(">", "").replace(".", " ").title()
    return name


def get_user_id(user_handle):
    if "|" in user_handle:
        name = user_handle.split("|")[0].replace("<", "").replace("@", "")
        return name

    return None


def render_rankings():
    message = "You must throw at least once to be ranked.\n"
    filtered_SNOWBALL_TABLE = {k: v for k, v in SNOWBALL_TABLE.items() if v["Attempt"] != 0}
    rankings_table_by_hit_success = "".join(
        [
            f"{key} Successful Hit: {value['Hit']} Attempts: {value['Attempt']}\n"
            for key, value in sorted(filtered_SNOWBALL_TABLE.items(), key=lambda item: item[1]["Hit"], reverse=True)
        ][:10]
    )

    rankings_table_by_hit_accuracy = "".join(
        [
            f"{key} Accuracy: {value['Hit']/value['Attempt']:.2f}\n"
            for key, value in sorted(
                filtered_SNOWBALL_TABLE.items(), key=lambda item: item[1]["Hit"] / item[1]["Attempt"], reverse=True
            )
        ][:10]
    )

    return render_message(message + rankings_table_by_hit_success + "\n\n" + rankings_table_by_hit_accuracy)


def render_stats(current_user):
    if SNOWBALL_TABLE[current_user]["Attempt"] == 0:
        return render_message("ERROR, DATA NOT FOUND \nAre we human? Or are we dancer?")

    accuracy = SNOWBALL_TABLE[current_user]["Hit"] / SNOWBALL_TABLE[current_user]["Attempt"]

    if accuracy > 0.9:
        message = f"Turn off your hacks or you will get nerfed!"
    elif accuracy > 0.5:
        message = f"You have a great throw! Keep it up!"
    elif accuracy > 0.3:
        message = f"You are alright. Statistically speaking you are very much just only alright."
    elif accuracy > 0.1:
        message = f"Probably work on your aim during your freetime."
    else:
        if SNOWBALL_TABLE[current_user]["Attempt"] > 10:
            message = f"Hey buddy, everything alright? Consider bribing someone... You probably need help."
        else:
            message = f"Keep trying! May the odds always be in your favor!"

    stat_table = (
        "\n"
        + "".join([f"{key}: {value}    " for key, value in SNOWBALL_TABLE[current_user].items()])
        + f"\n {current_user} has an accuracy of {accuracy:.2f}"
    )
    return render_message(message + stat_table)


def render_message(message):
    return jsonify({"response_type": "in_channel", "text": f"{message}"})


def snowball_outcomes(probability, current_user, target):
    SNOWBALL_TABLE[current_user]["Attempt"] += 1
    if SNOWBALL_TABLE[current_user]["Combo"] == 3:
        message = (
            f"You threw a snowball made of solid 24k gold. It hit {target} in the face and knocked them out for good."
        )
        message += f"\n"
        message += (
            f"It's honestly kind of messed up. {target} can't do anything snowball related for the next 5 minutes."
        )
        SNOWBALL_TABLE[target]["Stunned_Time"] = datetime.now()
        SNOWBALL_TABLE[current_user]["Hit"] += 1
        SNOWBALL_TABLE[current_user]["Combo"] = 0
        return message

    if probability < HIT_PROBABILITY:
        message = f"You hit {target} square in the back of the head. {target} is secretly crying right now."
        SNOWBALL_TABLE[current_user]["Hit"] += 1
        SNOWBALL_TABLE[current_user]["Combo"] += 1
        if SNOWBALL_TABLE[current_user]["Combo"] == 3:
            message += "\n"
            message += "You're on a streak! You get a golden snowball! Your next hit will prevent the person from throwing for 5 minutes!"  # noqa E501
    else:
        missing_probability = 1 - HIT_PROBABILITY
        SNOWBALL_TABLE[current_user]["Combo"] = 0
        if probability < (0.3 * missing_probability + HIT_PROBABILITY):
            message = f"You tripped and failed to hit your target, {target} is laughing at you from afar."
        elif probability < (0.6 * missing_probability + HIT_PROBABILITY):
            if target != "Stanley Phu" and current_user != "Stanley Phu":
                person_hit = "Stanley Phu"
            else:
                person_hit = "Yen-Ting Chen"
            message = f"You hit the ceiling, it bounces, and hits {person_hit} on the face instead. Try again maybe?"
        elif probability < (0.8 * missing_probability + HIT_PROBABILITY):
            message = f"You tried to hit {target} but hit the monitor instead. You may or may not have left a dent on that monitor."  # noqa E501
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
                dst_img = face_swap(
                    src_face, dst_face, src_points, dst_points, dst_shape, dst_img, warp_2d, correct_color
                )

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

            return (json_return, 200)
    return make_response(jsonify({"error": BAD_REQUEST}), 400)


def frisbee_outcomes(probability, target):
    success = False
    if probability < TOSS_PROBABILITY:
        message = f"1... 2... 3... You throw the disc at {target} and they make the catch! Great throw!"
        success = True
    else:
        if probability < 0.9:
            message = f"1... 2... You tripped and threw the disc straight into the ground. {target} and Cindy give you a death stare."  # noqa E501

        else:
            message = f"1... 2... 3... LEIASA, the impartial referee that she is, in her great wisdom, called a stall. WTF it’s only been 3 seconds."  # noqa E501

    return success, message


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

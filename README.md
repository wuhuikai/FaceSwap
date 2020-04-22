# FaceSwap
Swap face between two photos for Python 3 with OpenCV and dlib.

# Setup for developement:
- Setup a python 3.x venv (usually in `.venv`)
- `pip3 install --upgrade pip`
- Install dev requirements `pip3 install -r requirements.dev.txt`
- `pre-commit install`

# Run `pre-commit` locally.

`pre-commit run --all-files`

# Deploy with FrisbeeApp
to deploy with FrisbeeApp you need to make sure you include the slack api secret when you build

- `docker build --build-arg frisbee_token=${FRISBEE_TOKEN} -t YOURTAG .`

## Setup virtual environment (once)
virtualenv ./temp/env -p python3

# Source the virtual envirotnment (on every begining)
source ./temp/env/bin/activate

# install dependencies from requirements
pip3 install -r requirement.txt

# Keep the Models under below folders
./resources/

## Commands

# to download the images
python main.py --phase img_dl

# run pretrained mobilenet mode
python main.py --phase pretrained

# train the model
python main.py --phase train

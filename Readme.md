## Setup virtual environment (once)
virtualenv ./temp/env -p python3

# Source the virtual envirotnment (on every begining)
source ./temp/env/bin/activate

# Install dependencies from requirements
pip3 install -r requirement.txt

# Keep the Models under below folders
./resources/

## Commands

# Run pretrained mobilenet mode
python main.py --phase pretrained

# Run the custom binary classification after copying h5 model files into the ./resources folder
python main.py --phase test

## To train the Customer binary classification
# To download the binary image set
python main.py --phase img_dl

# Train the model. you might need a GPU. and then run the test
python main.py --phase train


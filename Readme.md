## setup virtual environment (once)
virtualenv ./temp/env -p python3

#source the virtual envirotnment (on every begining)
source ./temp/env/bin/activate

# install dependencies from requirements
pip3 install -r requirement.txt
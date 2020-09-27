# using python3 module "venv" to create a virtual-env name "venv"
python3 -m venv "venv"

# then activate it 
# win:
#    venv\Scripts\activate.bat
# unx:
source ./venv/bin/activate
pip -V
python -V

# update pip with pip
pip install --upgrade pip

# setup (w/ upgrade)
pip install --upgrade pandas
pip install --upgrade pandas_datareader
pip install --upgrade matplotlib


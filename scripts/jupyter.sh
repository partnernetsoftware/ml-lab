cd $HOME

pwd

# nohup jupyter lab --ip 0.0.0.0 --port 7777 --allow-root
nohup jupyter lab --ip 0.0.0.0 --port 7777 &

# setup a0 for jupyter-notebook password
nohup jupyter notebook --ip 0.0.0.0 --port 7776 &

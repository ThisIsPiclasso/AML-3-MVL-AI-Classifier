# How to run training
The training is ran through the train.py file, this has to be ran using 
```
uv run train.py
```

When ran like this the connection to the server needs to remain stable, otherwise training will stop.

To go around this we can spawn a 'remote' terminal on the server with 'tmux'
```
tmux new -s train_terminal
```
this will spawn a new terminal called train_terminal, and enter it.
You can run any command here, and call the training script.

to exit this remote terminal press Ctrl + B and afterwards press D and wait until you are kicked out of the terminal.
anything within the terminal will continue runnning.

to reenter the terminal run
```
tmux attach -t train_terminal
```


once done completely kill the remote terminal by running
```
tmux kill-session -t train_terminal
```

to check if a remote terminal is running. run
```
tmux ls
```
this will list all active remote terminals



# to run tensorboard
create a new terminal
```
tmux new -s train_monitor
```
```
uv run tensorboard --logdir=runs --host=0.0.0.0 --port=6006
```
press Ctrl + B and then D

The dashboard can be found on:
http://code-workspace:6006/

# Tuning

create terminal
```
tmux new -s tune_monitor
```
```
uvx optuna-dashboard sqlite:///optuna_tuning.db --host 0.0.0.0 --port 8080
```
Ctrl + B then D

create terminal
```
tmux new -s tune_terminal
```
```
uv run train.py
```
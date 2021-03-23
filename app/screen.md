# How to keep processes running after ending ssh session

- ssh into your remote box. type screen Then start the process you want.
- Press Ctrl-A then Ctrl-D. This will detach your screen session but leave your processes running. You can now log out of the remote box.
- If you want to come back later, log on again and type screen -r This will resume your screen session, and you can see the output of your process.

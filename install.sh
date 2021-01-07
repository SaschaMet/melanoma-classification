# chmod +x install.sh

echo "install everything ..."

# update pip
/usr/local/bin/python -m pip install --upgrade pip

# install requirements
pip install --user -r requirements.txt


# Optional: Install Kite (https://www.kite.com)

## 1. Install node
# wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.37.2/install.sh | bash
# nvm install 12

## 2. Install kite
# pip install jupyter-kite
# bash -c "$(wget -q -O - https://linux.kite.com/dls/linux/current)" -y
# jupyter labextension install "@kiteco/jupyterlab-kite"
# ~/.local/share/kite/login-user


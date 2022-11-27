import os
from pathlib import Path
import logging

os.makedirs("logs")

logging.basicConfig(filename="running_logs.log",
                    filemode="a",
                    level=logging.INFO,
                    format= "[%(asctime)s: %(levelname)s]: %(message)s"
                    )

while True:
    project_name = input("enter the project Name: ")
    if project_name != "":
        break

logging.info(f"creating project by name: {project_name}")

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"tests/__init__.py",
    "tests/unit/__init__.py",
    "tests/integration/__init__.py",
    "init_setup.sh",
    "requirements.txt",
    "requirements_dev.txt",
    "setup.py",
    "pyproject.toml",
    "setup.cfg",
    "tox.ini"
]

for filepath in list_of_files:
    filepath= Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok= True)
        logging.info(f"created a filedir : {filedir}")
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,"w") as f:
            pass
        logging.info(f"creating a new file : {filepath}")
    else:
        logging.info(f"file already exists at {filepath}")
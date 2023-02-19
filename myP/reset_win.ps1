#!/bin/bash
py manage.py runserver 0.0.0.0:8000 manage.py migrate
py runserver 0.0.0.0:8000 manage.py makemigrations images
py manage.py runserver 0.0.0.0:8000 manage.py runserver 0.0.0.0:8000
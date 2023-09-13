from django.db import models

class Users(models.Model):
    user_id = models.CharField(max_length=200)
    name = models.CharField(max_length=200)
    password = models.CharField(max_length=200)

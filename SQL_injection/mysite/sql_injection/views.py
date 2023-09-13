from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader

from sql_injection.models import Users

def index(request):
    template = loader.get_template('index.html')
    return HttpResponse(template.render({}, request))

def resetDB(request):
    Users.objects.all().delete()
    pathObj = Users(user_id="1", name="a", password="i")
    pathObj.save()
    pathObj = Users(user_id="2", name="b", password="ii")
    pathObj.save()
    pathObj = Users(user_id="3", name="c", password="iii")
    pathObj.save()
    return HttpResponseRedirect('/sql_injection/')

def fetch_with_id(request):
    user_id = request.POST.get('user_id')
    user_data = Users.objects.raw("SELECT * FROM sql_injection_users WHERE user_id = " + user_id)
    template = loader.get_template('show.html')
    context = {
        'user_data': user_data
    }
    return HttpResponse(template.render(context, request))

def login(request):
    username = request.POST.get('username')
    password = request.POST.get('password')
    user_data = Users.objects.raw('SELECT * FROM sql_injection_users WHERE name ="' + username + '" AND password ="' + password + '"')
    template = loader.get_template('show.html')
    context = {
        'user_data': user_data
    }
    return HttpResponse(template.render(context, request))


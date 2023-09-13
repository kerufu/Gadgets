from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path('resetDB/', views.resetDB, name='resetDB'),
    path('fetch_with_id/', views.fetch_with_id, name='fetch_with_id'),
    path('login/', views.login, name='login'),
]
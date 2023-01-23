from django.urls import URLPattern, path,include
from . import views
urlpatterns = [ 
    path('',views.home,name='home'),
    path('prediction_rice',views.prediction_rice,name='prediction_rice'),
    path('prediction_potato',views.prediction_potato,name='prediction_potato'),
    path('prediction_tomato',views.prediction_tomato,name='prediction_tomato'),
    path('prediction_cotton',views.prediction_cotton,name='prediction_cotton'),
]
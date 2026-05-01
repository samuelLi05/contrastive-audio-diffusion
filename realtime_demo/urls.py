from django.conf import settings
from django.conf.urls.static import static
from django.urls import path

from realtime_demo.demo import views


urlpatterns = [
    path("", views.index, name="demo-index"),
    path("api/status/", views.status, name="demo-status"),
    path("api/generate/", views.generate, name="demo-generate"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

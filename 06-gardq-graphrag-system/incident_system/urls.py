# incident_system/urls.py
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    # Inclure toutes les URLs de l'application incident_manager
    path('', include('incident_manager.urls')),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
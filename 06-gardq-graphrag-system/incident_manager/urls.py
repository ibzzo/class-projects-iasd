# incident_manager/urls.py
from django.urls import path
from .views import (
    IncidentSubmissionView, IncidentAnalysisView, 
    DataImportView, MetricsView, KGVisualizationView,
    KGDataView
)

urlpatterns = [
    path('', IncidentSubmissionView.as_view(), name='incident_submission'),
    path('analyze-incident/', IncidentAnalysisView.as_view(), name='analyze_incident'),
    path('import-data/', DataImportView.as_view(), name='import_data'),
    path('metrics/', MetricsView.as_view(), name='metrics'),
    path('api/metrics/', MetricsView.as_view(), name='api_metrics'),
    path('kg-visualization/', KGVisualizationView.as_view(), name='kg_visualization'),
    path('api/kg-data/', KGDataView.as_view(), name='kg_data'),
]
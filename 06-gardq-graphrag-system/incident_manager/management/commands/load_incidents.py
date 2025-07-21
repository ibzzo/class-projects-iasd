# incident_manager/management/commands/load_incidents.py
from django.core.management.base import BaseCommand
from incident_manager.data_loader import load_initial_data

class Command(BaseCommand):
    help = 'Charge les données initiales des incidents depuis le fichier CSV'

    def add_arguments(self, parser):
        parser.add_argument('file_path', type=str, help='Chemin vers le fichier CSV')

    def handle(self, *args, **kwargs):
        file_path = kwargs['file_path']
        result = load_initial_data(file_path)
        
        if result['success']:
            self.stdout.write(self.style.SUCCESS(result['message']))
        else:
            self.stdout.write(self.style.ERROR(result['message']))
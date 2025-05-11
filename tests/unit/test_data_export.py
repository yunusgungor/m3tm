"""
Data Export (Veri İndirme) modülü için birim testleri
"""
import os
import json
import tempfile
import pytest
import torch
import datetime
from pathlib import Path

from m3tm.data_export.export_manager import ExportManager
from m3tm.data_export.export_formats import (
    JSONExporter, 
    CSVExporter,
    TextExporter
)

class TestExportFormats:
    """Çeşitli dışa aktarma formatlarının testleri"""
    
    def test_json_exporter(self):
        """JSONExporter sınıfı testi"""
        # Test verisi
        data = [
            {"id": "doc1", "content": "Sample text 1", "metadata": {"type": "text", "created": "2024-01-01"}},
            {"id": "doc2", "content": "Sample text 2", "metadata": {"type": "text", "created": "2024-01-02"}},
            {"id": "img1", "content": "<binary_data>", "metadata": {"type": "image", "created": "2024-01-03"}}
        ]
        
        # Geçici dosya oluştur
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_filename = temp_file.name
        
        try:
            # JSONExporter kullanarak dışa aktar
            exporter = JSONExporter()
            exporter.export(data, temp_filename)
            
            # Dosyayı oku ve doğrula
            with open(temp_filename, 'r') as f:
                loaded_data = json.load(f)
            
            assert len(loaded_data) == 3
            assert loaded_data[0]["id"] == "doc1"
            assert loaded_data[1]["id"] == "doc2"
            assert loaded_data[2]["metadata"]["type"] == "image"
        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
    
    def test_csv_exporter(self):
        """CSVExporter sınıfı testi"""
        # Test verisi
        data = [
            {"id": "doc1", "content": "Sample text 1", "created": "2024-01-01"},
            {"id": "doc2", "content": "Sample text 2", "created": "2024-01-02"},
            {"id": "doc3", "content": "Sample text 3", "created": "2024-01-03"}
        ]
        
        # Geçici dosya oluştur
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
            temp_filename = temp_file.name
        
        try:
            # CSVExporter kullanarak dışa aktar
            exporter = CSVExporter()
            exporter.export(data, temp_filename)
            
            # Dosyanın varlığını doğrula
            assert os.path.exists(temp_filename)
            
            # Dosya içeriğini kontrol et
            with open(temp_filename, 'r') as f:
                lines = f.readlines()
            
            # Başlık ve 3 satır veri bekliyoruz
            assert len(lines) == 4
            
            # Başlıkları kontrol et (sıralama garanti değil)
            headers = lines[0].strip().split(',')
            assert 'id' in headers
            assert 'content' in headers
            assert 'created' in headers
        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
    
    def test_text_exporter(self):
        """TextExporter sınıfı testi"""
        # Test verisi
        data = [
            {"id": "doc1", "content": "Sample text 1", "metadata": {"type": "text"}},
            {"id": "doc2", "content": "Sample text 2", "metadata": {"type": "text"}},
        ]
        
        # Geçici dosya oluştur
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_filename = temp_file.name
        
        try:
            # TextExporter kullanarak dışa aktar
            exporter = TextExporter()
            exporter.export(data, temp_filename)
            
            # Dosya içeriğini kontrol et
            with open(temp_filename, 'r') as f:
                content = f.read()
            
            # Her içerik metni dosyada olmalı
            assert "Sample text 1" in content
            assert "Sample text 2" in content
        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

class TestExportManager:
    """ExportManager sınıfı testleri"""
    
    def test_export_manager_initialization(self):
        """ExportManager başlatma testi"""
        manager = ExportManager()
        
        # ExportManager'ın format desteklerini kontrol et
        assert "json" in manager.supported_formats
        assert "csv" in manager.supported_formats
        assert "txt" in manager.supported_formats
    
    def test_export_manager_format_detection(self):
        """ExportManager format algılama testi"""
        manager = ExportManager()
        
        # Dosya uzantısından format algılama
        assert manager._detect_format("data.json") == "json"
        assert manager._detect_format("data.csv") == "csv"
        assert manager._detect_format("data.txt") == "txt"
        
        # Bilinmeyen format
        with pytest.raises(ValueError):
            manager._detect_format("data.xyz")
    
    def test_export_manager_export(self, mocker):
        """ExportManager dışa aktarma testi"""
        manager = ExportManager()
        
        # Mocklamak için dışa aktarma sınıflarını izle
        mock_json_export = mocker.patch.object(JSONExporter, 'export')
        mock_csv_export = mocker.patch.object(CSVExporter, 'export')
        
        # Test verisi
        data = [{"id": "doc1", "content": "test"}]
        
        # JSON dışa aktarma
        manager.export(data, "output.json")
        mock_json_export.assert_called_once()
        
        # CSV dışa aktarma
        manager.export(data, "output.csv")
        mock_csv_export.assert_called_once()
        
        # Format belirterek dışa aktarma
        mock_json_export.reset_mock()
        manager.export(data, "output.xyz", format="json")
        mock_json_export.assert_called_once()
    
    def test_export_manager_with_date_filter(self, mocker):
        """ExportManager tarih filtreleme testi"""
        manager = ExportManager()
        
        # Mock exporter
        mock_exporter = mocker.Mock()
        mocker.patch.object(manager, '_get_exporter', return_value=mock_exporter)
        
        # Test verisi
        data = [
            {
                "id": "doc1", 
                "content": "Sample text 1", 
                "metadata": {"created": "2024-01-01T10:00:00Z"}
            },
            {
                "id": "doc2", 
                "content": "Sample text 2", 
                "metadata": {"created": "2024-02-01T10:00:00Z"}
            },
            {
                "id": "doc3", 
                "content": "Sample text 3", 
                "metadata": {"created": "2024-03-01T10:00:00Z"}
            }
        ]
        
        # Tarih filtreleri
        start_date = datetime.datetime(2024, 2, 1, 0, 0, 0)
        end_date = datetime.datetime(2024, 2, 28, 23, 59, 59)
        
        # Filtreyle dışa aktarma
        manager.export(data, "output.json", start_date=start_date, end_date=end_date)
        
        # Filtrelemenin doğru veriyi geçirdiğini kontrol et
        # Sadece 'doc2' geçmelidir
        exported_data = mock_exporter.export.call_args[0][0]
        assert len(exported_data) == 1
        assert exported_data[0]["id"] == "doc2"
    
    def test_export_manager_with_type_filter(self, mocker):
        """ExportManager tip filtreleme testi"""
        manager = ExportManager()
        
        # Mock exporter
        mock_exporter = mocker.Mock()
        mocker.patch.object(manager, '_get_exporter', return_value=mock_exporter)
        
        # Test verisi
        data = [
            {"id": "doc1", "content": "Sample text 1", "metadata": {"type": "text"}},
            {"id": "img1", "content": "<binary>", "metadata": {"type": "image"}},
            {"id": "doc2", "content": "Sample text 2", "metadata": {"type": "text"}}
        ]
        
        # Tipe göre filtreleme
        manager.export(data, "output.json", content_type="text")
        
        # Filtrelemenin doğru veriyi geçirdiğini kontrol et
        # Sadece 'doc1' ve 'doc2' geçmelidir
        exported_data = mock_exporter.export.call_args[0][0]
        assert len(exported_data) == 2
        assert exported_data[0]["id"] == "doc1"
        assert exported_data[1]["id"] == "doc2"
    
    def test_export_manager_with_limit(self, mocker):
        """ExportManager limit testi"""
        manager = ExportManager()
        
        # Mock exporter
        mock_exporter = mocker.Mock()
        mocker.patch.object(manager, '_get_exporter', return_value=mock_exporter)
        
        # Test verisi
        data = [
            {"id": "doc1", "content": "Sample text 1"},
            {"id": "doc2", "content": "Sample text 2"},
            {"id": "doc3", "content": "Sample text 3"},
            {"id": "doc4", "content": "Sample text 4"},
            {"id": "doc5", "content": "Sample text 5"}
        ]
        
        # Limitle dışa aktarma
        manager.export(data, "output.json", limit=3)
        
        # Filtrelemenin doğru veriyi geçirdiğini kontrol et
        # Sadece ilk 3 belge geçmelidir
        exported_data = mock_exporter.export.call_args[0][0]
        assert len(exported_data) == 3
        assert exported_data[0]["id"] == "doc1"
        assert exported_data[1]["id"] == "doc2"
        assert exported_data[2]["id"] == "doc3" 
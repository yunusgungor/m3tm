"""
M³TM modelinin arama modülü sağlamlık testleri

Bu modül, arama bileşeninin çeşitli zorlu durumlardaki davranışını test eder:
- Büyük indekslerle arama performansı
- Semantik olarak benzer ama farklı anahtar kelimelerle arama
- Multimodal aramaların tutarlılığı
- Yazım hataları ve gürültülü sorgularla arama
- Çok dilli aramaların davranışı
"""
import pytest
import torch
import numpy as np
import tempfile
import os
import time
from pathlib import Path
from unittest.mock import patch

from m3tm.config.model_config import get_tiny_config
from m3tm.core.base_model import M3TMBaseModel
from m3tm.search.search_service import SearchService
from m3tm.search.search_embedding import SearchEmbeddingProjection


class TestSearchRobustness:
    """M³TM modelinin arama modülü sağlamlık testleri"""
    
    def setup_model_and_search(self):
        """Test için model ve arama servisi oluşturur"""
        config = get_tiny_config()
        model = M3TMBaseModel(config)
        
        # SearchService örneklemesi
        search_service = SearchService(config.search_config.search_dim)
        
        # Kolay test için mock embedding fonksiyonu hazırla
        def mock_get_embedding(text=None, image=None):
            embed_dim = config.search_config.search_dim
            if text is not None:
                # Metinden basit bir embedding oluştur
                # (gerçek bir model olayı metin içeriğine göre değiştirecektir)
                import hashlib
                text_hash = hashlib.md5(text.encode()).digest()
                seed = int.from_bytes(text_hash[:4], byteorder='big')
                np.random.seed(seed)
                embedding = np.random.randn(embed_dim)
            elif image is not None:
                # Görüntüden basit bir embedding oluştur
                # (gerçek bir model bunu görüntü içeriğine göre değiştirecektir)
                if isinstance(image, torch.Tensor):
                    image_sum = float(image.sum().item())
                else:
                    image_sum = float(np.sum(image))
                np.random.seed(int(image_sum * 1000) % 2**32)
                embedding = np.random.randn(embed_dim)
            else:
                raise ValueError("Metin veya görüntü girişi gereklidir")
                
            # Normalize et
            embedding = embedding / np.linalg.norm(embedding)
            return torch.tensor(embedding, dtype=torch.float32)
            
        # Mock fonksiyonu yerleştir
        search_service._get_embedding = mock_get_embedding
        
        return model, search_service, config
    
    def generate_sample_content(self, count=100):
        """Test için örnek içerik oluşturur"""
        sample_texts = []
        sample_metadata = []
        
        categories = ["politika", "ekonomi", "spor", "teknoloji", "sağlık", "eğitim", "sanat", "bilim"]
        
        for i in range(count):
            category = categories[i % len(categories)]
            # Her kategoriye özgü içerik oluştur
            if category == "politika":
                text = f"Politika haberi {i}: Yerel seçimler, hükümet kararları ve uluslararası ilişkiler hakkında güncel bilgiler."
            elif category == "ekonomi":
                text = f"Ekonomi analizi {i}: Döviz kurları, borsa hareketleri, enflasyon oranları ve ekonomik politikalar."
            elif category == "spor":
                text = f"Spor haberi {i}: Futbol maçları, basketbol turnuvaları ve diğer spor etkinlikleri hakkında son dakika gelişmeleri."
            elif category == "teknoloji":
                text = f"Teknoloji incelemesi {i}: Yeni akıllı telefonlar, yazılım güncellemeleri ve teknoloji şirketlerinin ürünleri."
            elif category == "sağlık":
                text = f"Sağlık rehberi {i}: Sağlıklı yaşam ipuçları, beslenme tavsiyeleri ve hastalıklardan korunma yöntemleri."
            elif category == "eğitim":
                text = f"Eğitim raporu {i}: Üniversite sınavları, yeni eğitim politikaları ve okul sistemindeki değişiklikler."
            elif category == "sanat":
                text = f"Sanat etkinliği {i}: Sergiler, konserler, tiyatro gösterileri ve sanat dünyasından haberler."
            elif category == "bilim":
                text = f"Bilimsel gelişme {i}: Yeni araştırmalar, bilimsel keşifler ve uzay bilimleri hakkında güncel bilgiler."
            
            sample_texts.append(text)
            sample_metadata.append({
                "id": f"doc_{i}",
                "type": "text",
                "category": category,
                "date": f"2023-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}"
            })
            
        return sample_texts, sample_metadata
    
    @pytest.mark.search
    def test_large_index_performance(self):
        """Büyük indeksle arama performansı testi"""
        _, search_service, _ = self.setup_model_and_search()
        
        # Farklı boyutlarda indeksler için performans testleri
        index_sizes = [100, 500, 1000, 5000]
        if pytest.config.getoption("--runslow", default=False):
            index_sizes.extend([10000, 50000, 100000])
        
        results = {}
        
        for size in index_sizes:
            # Örnek içerik oluştur
            sample_texts, sample_metadata = self.generate_sample_content(size)
            
            # Zaman ölçümü başlat
            start_index_time = time.time()
            
            # İçeriği ekle
            for text, metadata in zip(sample_texts, sample_metadata):
                search_service.add_content(text=text, metadata=metadata)
            
            # İndeksleme süresini hesapla
            index_time = time.time() - start_index_time
            
            # Birkaç test sorgusu
            test_queries = [
                "politika seçim haberleri", 
                "ekonomi döviz kurları", 
                "spor futbol maçları",
                "teknoloji akıllı telefonlar"
            ]
            
            search_times = []
            
            for query in test_queries:
                # Arama zamanını ölç
                start_search_time = time.time()
                results = search_service.search(text=query, top_k=10)
                search_time = time.time() - start_search_time
                search_times.append(search_time)
            
            # Ortalama arama süresini hesapla
            avg_search_time = sum(search_times) / len(search_times)
            
            # Sonuçları kaydet
            results[size] = {
                "index_time": index_time,
                "avg_search_time": avg_search_time
            }
            
            print(f"\nİndeks boyutu: {size}")
            print(f"İndeksleme süresi: {index_time:.4f} saniye")
            print(f"Ortalama arama süresi: {avg_search_time:.4f} saniye")
            
            # Büyük indeksler için performans eşikleri
            # Not: Gerçek eşikler donanıma ve uygulama gereksinimlerine bağlıdır
            assert avg_search_time < 0.05, f"{size} boyutunda indeks için arama süresi çok uzun: {avg_search_time:.4f} saniye"
    
    @pytest.mark.search
    def test_semantic_similarity_robustness(self):
        """Semantik benzerlik sağlamlığı testi"""
        _, search_service, _ = self.setup_model_and_search()
        
        # Örnek içerik oluştur ve indeksle
        sample_texts = [
            "Akıllı telefonların batarya ömrünü uzatma yöntemleri ve enerji tasarrufu ipuçları",
            "Türkiye'de enflasyon oranları ve ekonomik göstergeler hakkında detaylı analiz",
            "İstanbul'da bu hafta sonu düzenlenecek konserler ve kültürel etkinlikler",
            "Kalp sağlığını korumak için beslenme önerileri ve yaşam tarzı değişiklikleri",
            "Yeni çıkan elektrikli araçlar ve otomotiv sektöründeki son teknolojik gelişmeler",
            "Milli futbol takımımızın kadrosu ve yaklaşan dünya kupası elemelerindeki şansı",
            "Yazılım geliştirme süreçleri ve modern yazılım projelerinde kullanılan metodolojiler",
            "Üniversite sınavları için çalışma teknikleri ve verimli ders çalışma yöntemleri"
        ]
        
        for i, text in enumerate(sample_texts):
            search_service.add_content(
                text=text,
                metadata={
                    "id": f"doc_{i}",
                    "type": "text",
                    "title": f"Doküman {i}"
                }
            )
        
        # Semantik olarak benzer ama farklı kelimeler içeren sorgular
        semantic_query_pairs = [
            # Telefon batarya ilgili sorgular
            ("Cep telefonu şarjı nasıl uzun sürer?", "Mobil cihazlarda pil ömrünü artırma"),
            # Ekonomiyle ilgili sorgular
            ("Türkiye ekonomisi ne durumda?", "Ekonomik göstergeler ve enflasyon analizi"),
            # Konserlerle ilgili sorgular
            ("İstanbul etkinlik takvimi", "İstanbul'daki müzik performansları"),
            # Sağlıkla ilgili sorgular
            ("Kardiyovasküler sağlık için öneriler", "Kalp hastalıklarından korunma yolları"),
            # Elektrikli araçlarla ilgili sorgular
            ("Elektrikli araba modelleri ve özellikleri", "Çevre dostu otomobil teknolojileri")
        ]
        
        # Her sorgu çifti için, semantik olarak benzer sonuçlar döndürmeli
        for query1, query2 in semantic_query_pairs:
            # İlk sorgu sonuçları
            results1 = search_service.search(text=query1, top_k=3)
            # İkinci sorgu sonuçları
            results2 = search_service.search(text=query2, top_k=3)
            
            # Sonuçlar arasında benzerlik olmalı (en az bir öğe ortak)
            ids1 = [r["metadata"]["id"] for r in results1]
            ids2 = [r["metadata"]["id"] for r in results2]
            
            common_items = set(ids1).intersection(set(ids2))
            
            print(f"\nSorgu 1: '{query1}'")
            print(f"Sorgu 2: '{query2}'")
            print(f"Ortak sonuçlar: {common_items}")
            
            assert len(common_items) > 0, f"Semantik olarak benzer sorgular farklı sonuçlar döndürdü: '{query1}' ve '{query2}'"
    
    @pytest.mark.search
    def test_spelling_errors_robustness(self):
        """Yazım hataları ve gürültülü sorgu sağlamlığı testi"""
        _, search_service, _ = self.setup_model_and_search()
        
        # Örnek içerik oluştur ve indeksle
        sample_texts = [
            "Akıllı telefonlar için uygulama önerileri ve en iyi uygulamalar listesi",
            "İzmir'de gezilecek yerler ve turistik mekanlar hakkında rehber",
            "Yazılım mühendisliği ve programlama dilleri üzerine kapsamlı değerlendirme",
            "Türk mutfağından lezzetli tarifler ve yemek pişirme teknikleri",
            "Doğa yürüyüşü ve kamp yapmak için en iyi rotalar ve ekipman önerileri"
        ]
        
        for i, text in enumerate(sample_texts):
            search_service.add_content(
                text=text,
                metadata={
                    "id": f"doc_{i}",
                    "type": "text",
                    "title": f"Doküman {i}"
                }
            )
        
        # Doğru sorgular ve yazım hatalı karşılıkları
        query_pairs = [
            ("akıllı telefon uygulamaları", "akılı telafon uygulmalrı"),
            ("izmir'de gezilecek yerler", "izimrde gezilcek yeler"),
            ("yazılım programlama", "yazlım proramlama"),
            ("türk mutfağı tarifler", "türk mutfğı tarfiler"),
            ("doğa yürüyüşü ve kamp", "doğ yürüyşü ve kamp")
        ]
        
        for correct_query, misspelled_query in query_pairs:
            # Doğru sorgu sonuçları
            correct_results = search_service.search(text=correct_query, top_k=2)
            # Yazım hatalı sorgu sonuçları
            misspelled_results = search_service.search(text=misspelled_query, top_k=2)
            
            # Sonuçlar arasında benzerlik olmalı
            correct_ids = [r["metadata"]["id"] for r in correct_results]
            misspelled_ids = [r["metadata"]["id"] for r in misspelled_results]
            
            print(f"\nDoğru sorgu: '{correct_query}'")
            print(f"Hatalı sorgu: '{misspelled_query}'")
            print(f"Doğru sonuçlar: {correct_ids}")
            print(f"Hatalı sonuçlar: {misspelled_ids}")
            
            # En az bir ortak sonuç olmalı
            common_items = set(correct_ids).intersection(set(misspelled_ids))
            assert len(common_items) > 0, f"Yazım hatalı sorgu tamamen farklı sonuçlar verdi: '{correct_query}' vs '{misspelled_query}'"
    
    @pytest.mark.search
    def test_multilingual_search(self):
        """Çok dilli arama sağlamlığı testi"""
        _, search_service, _ = self.setup_model_and_search()
        
        # Farklı dillerde örnek içerik
        multilingual_texts = [
            # Türkçe
            "Türkiye'nin en güzel plajları ve tatil yerleri hakkında kapsamlı rehber",
            # İngilizce karşılığı
            "Comprehensive guide to the most beautiful beaches and vacation spots in Turkey",
            # Almanca
            "Umfassender Leitfaden zu den schönsten Stränden und Ferienorten der Türkei",
            # Fransızca
            "Guide complet des plus belles plages et lieux de vacances en Turquie",
            # İspanyolca
            "Guía completa de las playas más hermosas y lugares de vacaciones en Turquía"
        ]
        
        # İçerikleri ekle
        for i, text in enumerate(multilingual_texts):
            search_service.add_content(
                text=text,
                metadata={
                    "id": f"doc_{i}",
                    "type": "text",
                    "language": ["tr", "en", "de", "fr", "es"][i]
                }
            )
        
        # Farklı dillerde sorgular
        multilingual_queries = [
            ("türkiye plaj tatil", "tr"),
            ("turkey beach vacation", "en"),
            ("türkei strand urlaub", "de"),
            ("turquie plage vacances", "fr"),
            ("turquía playa vacaciones", "es")
        ]
        
        all_results = {}
        
        # Her dildeki sorguyu test et
        for query, lang in multilingual_queries:
            results = search_service.search(text=query, top_k=5)
            result_ids = [r["metadata"]["id"] for r in results]
            all_results[lang] = result_ids
            
            # Her sorgu en azından kendi dilindeki içeriği bulmalı
            lang_index = ["tr", "en", "de", "fr", "es"].index(lang)
            expected_doc_id = f"doc_{lang_index}"
            
            print(f"\nDil: {lang}, Sorgu: '{query}'")
            print(f"Sonuçlar: {result_ids}")
            
            assert expected_doc_id in result_ids, f"{lang} dilindeki sorgu kendi dilindeki içeriği bulamadı"
        
        # Her dilin sonuçlarını diğerleriyle karşılaştır
        # Semantik olarak benzer olduklarından, bazı ortak sonuçlar olmalı
        for lang1 in ["tr", "en", "de", "fr", "es"]:
            for lang2 in ["tr", "en", "de", "fr", "es"]:
                if lang1 != lang2:
                    common = set(all_results[lang1]).intersection(set(all_results[lang2]))
                    print(f"Dil çifti {lang1}-{lang2}, Ortak sonuçlar: {common}")
                    
                    # En az bir ortak sonuç olmalı (aynı içeriğin çevirileri)
                    assert len(common) > 0, f"Dil çifti {lang1}-{lang2} arasında ortak sonuç yok"
    
    @pytest.mark.search
    def test_multimodal_search_consistency(self):
        """Multimodal arama tutarlılığı testi"""
        _, search_service, _ = self.setup_model_and_search()
        
        # Metin-görüntü çiftleri oluştur (bu örnekte görüntüler dummy veriler)
        # Gerçek testlerde görüntüler anlamlı olacaktır
        sample_contents = [
            {
                "text": "Kırmızı bir spor araba, yüksek hızda otoyolda gidiyor",
                "image": np.random.rand(64, 64, 3) * 0.5 + 0.5,  # Kırmızımsı gölgeler
                "metadata": {"id": "item_1", "category": "araba"}
            },
            {
                "text": "Mavi deniz kenarında palmiye ağaçları ve kumsalda şemsiyeler",
                "image": np.random.rand(64, 64, 3) * 0.3 + np.array([0.0, 0.0, 0.7]),  # Mavimsi gölgeler
                "metadata": {"id": "item_2", "category": "plaj"}
            },
            {
                "text": "Dağ manzarası, karlı zirveler ve yeşil vadiler",
                "image": np.random.rand(64, 64, 3) * 0.3 + np.array([0.0, 0.5, 0.0]),  # Yeşilimsi gölgeler
                "metadata": {"id": "item_3", "category": "dağ"}
            },
            {
                "text": "Modern bir şehir manzarası, gökdelenler ve kalabalık caddeler",
                "image": np.random.rand(64, 64, 3) * 0.5 + 0.25,  # Gri tonlar
                "metadata": {"id": "item_4", "category": "şehir"}
            }
        ]
        
        # İçeriği ekle
        for item in sample_contents:
            search_service.add_content(
                text=item["text"],
                image=item["image"],
                metadata=item["metadata"]
            )
        
        # Test senaryosu: Yalnızca metin, yalnızca görüntü ve her ikisiyle arama
        test_scenarios = [
            {"name": "Araba - Yalnızca Metin", "text": "kırmızı spor araba", "image": None, "expected_id": "item_1"},
            {"name": "Plaj - Yalnızca Metin", "text": "deniz kenarı kumsal", "image": None, "expected_id": "item_2"},
            
            # Görüntü arama senaryoları - orijinal görüntünün hafifçe değiştirilmiş hali
            {
                "name": "Araba - Yalnızca Görüntü", 
                "text": None, 
                "image": sample_contents[0]["image"] * 0.9 + 0.05,  # Hafif değişiklik
                "expected_id": "item_1"
            },
            {
                "name": "Plaj - Yalnızca Görüntü", 
                "text": None, 
                "image": sample_contents[1]["image"] * 0.9 + 0.05,  # Hafif değişiklik
                "expected_id": "item_2"
            },
            
            # Multimodal arama senaryoları
            {
                "name": "Araba - Multimodal", 
                "text": "hızlı araba", 
                "image": sample_contents[0]["image"] * 0.9 + 0.05,  # Hafif değişiklik
                "expected_id": "item_1"
            },
            {
                "name": "Plaj - Multimodal", 
                "text": "kumsal tatil", 
                "image": sample_contents[1]["image"] * 0.9 + 0.05,  # Hafif değişiklik
                "expected_id": "item_2"
            },
        ]
        
        # Her senaryoyu test et
        for scenario in test_scenarios:
            results = search_service.search(
                text=scenario["text"], 
                image=scenario["image"], 
                top_k=1
            )
            
            print(f"\nSenaryo: {scenario['name']}")
            print(f"Sonuç: {results[0]['metadata']['id'] if results else 'Sonuç yok'}")
            
            assert len(results) > 0, f"{scenario['name']} senaryosu için sonuç döndürülmedi"
            assert results[0]["metadata"]["id"] == scenario["expected_id"], \
                f"{scenario['name']} senaryosu için beklenen sonuç {scenario['expected_id']} ama {results[0]['metadata']['id']} döndü" 
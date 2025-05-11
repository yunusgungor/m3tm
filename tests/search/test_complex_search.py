"""
M³TM modelinin karmaşık arama senaryoları testleri

Bu modül, arama bileşeninin gelişmiş kullanım senaryolarını test eder:
- Hibrit metin-görüntü aramaları
- Filtreli ve sıralı aramalar
- Arama özelleştirme ve ayarları
- Arama sonuçları birleştirme
- Arama sonuçları önbelleği
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
from m3tm.search.hybrid_search import HybridSearchEngine


class TestComplexSearch:
    """M³TM modelinin karmaşık arama senaryoları testleri"""
    
    def setup_model_and_search(self, use_hybrid=False):
        """Test için model ve arama servisi oluşturur"""
        config = get_tiny_config()
        model = M3TMBaseModel(config)
        
        if use_hybrid:
            # Hibrit arama motoru oluştur
            search_engine = HybridSearchEngine(
                embedding_dim=config.search_config.search_dim,
                use_text_index=True,
                use_vector_index=True
            )
        else:
            # Standart arama servisi
            search_engine = SearchService(config.search_config.search_dim)
        
        return model, search_engine, config
    
    def generate_test_data(self, count=50):
        """Test için yapay veri oluşturur"""
        # Metin-görüntü çiftleri ve zengin metadata
        text_samples = []
        image_samples = []
        metadata_samples = []
        
        # Farklı kategorilerde içerik oluştur
        categories = ["haber", "blog", "ürün", "sosyal", "akademik"]
        languages = ["tr", "en", "de", "fr", "es"]
        date_range = [f"2023-{month:02d}-{day:02d}" for month in range(1, 13) for day in range(1, 29, 7)]
        
        for i in range(count):
            # Kategori ve dil seç
            category = categories[i % len(categories)]
            language = languages[i % len(languages)]
            date = date_range[i % len(date_range)]
            
            # İçerik oluştur
            if category == "haber":
                text = f"Güncel haber {i}: Politik gelişmeler, ekonomi haberleri ve dünya gündemi."
                # Haber renkleri (genellikle mavi tonları)
                image = np.random.rand(64, 64, 3) * 0.3 + np.array([0.1, 0.1, 0.5])
            elif category == "blog":
                text = f"Blog yazısı {i}: Kişisel deneyimler, yaşam tarzı ipuçları ve günlük düşünceler."
                # Blog renkleri (genellikle yeşil tonları)
                image = np.random.rand(64, 64, 3) * 0.3 + np.array([0.1, 0.5, 0.1])
            elif category == "ürün":
                text = f"Ürün tanıtımı {i}: En yeni elektronik cihazlar, giyim ürünleri ve ev eşyaları."
                # Ürün renkleri (genellikle kırmızı tonları)
                image = np.random.rand(64, 64, 3) * 0.3 + np.array([0.5, 0.1, 0.1])
            elif category == "sosyal":
                text = f"Sosyal medya gönderisi {i}: Etkinlikler, arkadaş buluşmaları ve topluluk aktiviteleri."
                # Sosyal medya renkleri (genellikle mor tonları)
                image = np.random.rand(64, 64, 3) * 0.3 + np.array([0.3, 0.1, 0.5])
            else:  # akademik
                text = f"Akademik makale {i}: Bilimsel araştırmalar, teorik çalışmalar ve literatür incelemeleri."
                # Akademik renkleri (genellikle gri tonları)
                image = np.random.rand(64, 64, 3) * 0.3 + 0.3
            
            # Metadata oluştur
            likes = np.random.randint(0, 1000) if category in ["blog", "sosyal"] else 0
            views = np.random.randint(100, 10000)
            
            metadata = {
                "id": f"doc_{i}",
                "type": "multimodal",
                "category": category,
                "language": language,
                "date": date,
                "likes": likes,
                "views": views,
                "priority": np.random.randint(1, 6),  # 1-5 arası öncelik
                "tags": [f"tag_{j}" for j in range(1, 4) if (i + j) % 5 == 0]  # Rastgele etiketler
            }
            
            text_samples.append(text)
            image_samples.append(image)
            metadata_samples.append(metadata)
        
        return text_samples, image_samples, metadata_samples
    
    @pytest.mark.search
    def test_hybrid_text_image_search(self):
        """Hibrit metin-görüntü aramaları testi"""
        model, search_engine, _ = self.setup_model_and_search(use_hybrid=True)
        
        # Test verisi hazırla ve ekle
        text_samples, image_samples, metadata_samples = self.generate_test_data(count=30)
        
        for text, image, metadata in zip(text_samples, image_samples, metadata_samples):
            search_engine.add_content(text=text, image=image, metadata=metadata)
        
        # Hibrit arama senaryoları
        test_scenarios = [
            {
                "name": "Yalnızca Metin",
                "query_text": "güncel haber politika",
                "query_image": None,
                "expected_category": "haber",
                "top_k": 3
            },
            {
                "name": "Yalnızca Görüntü",
                "query_text": None,
                "query_image": np.random.rand(64, 64, 3) * 0.3 + np.array([0.1, 0.1, 0.5]),  # Haber benzeri renk
                "expected_category": "haber",
                "top_k": 3
            },
            {
                "name": "Metin ve Görüntü",
                "query_text": "elektronik ürünler",
                "query_image": np.random.rand(64, 64, 3) * 0.3 + np.array([0.5, 0.1, 0.1]),  # Ürün benzeri renk
                "expected_category": "ürün",
                "top_k": 3
            }
        ]
        
        # Her senaryoyu test et
        for scenario in test_scenarios:
            results = search_engine.search(
                text=scenario["query_text"],
                image=scenario["query_image"],
                top_k=scenario["top_k"]
            )
            
            # Yeterli sonuç döndürülmeli
            assert len(results) > 0, f"{scenario['name']} senaryosu sonuç döndürmedi"
            
            # Sonuçlar arasında beklenen kategoriden en az bir sonuç olmalı
            categories = [result["metadata"]["category"] for result in results]
            
            print(f"\nSenaryo: {scenario['name']}")
            print(f"Sorgu: Metin: {scenario['query_text']}, Görüntü: {'Var' if scenario['query_image'] is not None else 'Yok'}")
            print(f"Sonuç kategorileri: {categories}")
            
            assert scenario["expected_category"] in categories, \
                f"{scenario['name']} senaryosu {scenario['expected_category']} kategorisinde sonuç döndürmedi"
    
    @pytest.mark.search
    def test_filtered_and_sorted_search(self):
        """Filtreli ve sıralı aramalar testi"""
        _, search_engine, _ = self.setup_model_and_search()
        
        # Test verisi hazırla ve ekle
        text_samples, image_samples, metadata_samples = self.generate_test_data(count=50)
        
        for text, image, metadata in zip(text_samples, image_samples, metadata_samples):
            search_engine.add_content(text=text, image=image, metadata=metadata)
        
        # Filtre ve sıralama için test senaryoları
        test_scenarios = [
            {
                "name": "Kategori Filtresi",
                "query_text": "makale araştırma",
                "filters": {"category": "akademik"},
                "sort_by": None,
                "top_k": 5
            },
            {
                "name": "Dil Filtresi",
                "query_text": "haber gündem",
                "filters": {"language": "tr"},
                "sort_by": None,
                "top_k": 5
            },
            {
                "name": "Tarih Filtresi",
                "query_text": "güncel haberler",
                "filters": {"date": "2023-05-01"},  # Belirli bir tarih
                "sort_by": None,
                "top_k": 5
            },
            {
                "name": "Görüntülenme Sıralaması",
                "query_text": "popüler içerik",
                "filters": {},
                "sort_by": {"field": "views", "order": "desc"},
                "top_k": 5
            },
            {
                "name": "Beğeni Sıralaması",
                "query_text": "beğenilen gönderiler",
                "filters": {"category": "sosyal"},
                "sort_by": {"field": "likes", "order": "desc"},
                "top_k": 5
            },
            {
                "name": "Tarih Sıralaması",
                "query_text": "son haberler",
                "filters": {"category": "haber"},
                "sort_by": {"field": "date", "order": "desc"},
                "top_k": 5
            },
            {
                "name": "Çoklu Filtre ve Sıralama",
                "query_text": "önemli içerik",
                "filters": {"language": "tr", "priority": 5},
                "sort_by": {"field": "views", "order": "desc"},
                "top_k": 5
            }
        ]
        
        # Her senaryoyu test et
        for scenario in test_scenarios:
            results = search_engine.search(
                text=scenario["query_text"],
                top_k=scenario["top_k"],
                filters=scenario["filters"],
                sort_by=scenario["sort_by"]
            )
            
            print(f"\nSenaryo: {scenario['name']}")
            print(f"Filtreler: {scenario['filters']}")
            print(f"Sıralama: {scenario['sort_by']}")
            print(f"Sonuç Sayısı: {len(results)}")
            
            # Filtre sonuçlarını doğrula
            if scenario["filters"] and len(results) > 0:
                for key, value in scenario["filters"].items():
                    for result in results:
                        assert result["metadata"][key] == value, \
                            f"{scenario['name']} senaryosunda filtre sorunu: {key}={value}"
            
            # Sıralama sonuçlarını doğrula
            if scenario["sort_by"] and len(results) > 1:
                field = scenario["sort_by"]["field"]
                is_desc = scenario["sort_by"]["order"] == "desc"
                
                for i in range(len(results) - 1):
                    current = results[i]["metadata"][field]
                    next_item = results[i + 1]["metadata"][field]
                    
                    if is_desc:
                        assert current >= next_item, \
                            f"{scenario['name']} senaryosunda azalan sıralama sorunu: {current} < {next_item}"
                    else:
                        assert current <= next_item, \
                            f"{scenario['name']} senaryosunda artan sıralama sorunu: {current} > {next_item}"
    
    @pytest.mark.search
    def test_search_customization(self):
        """Arama özelleştirme ve ayarları testi"""
        _, search_engine, _ = self.setup_model_and_search()
        
        # Test verisi hazırla ve ekle
        text_samples, image_samples, metadata_samples = self.generate_test_data(count=40)
        
        for text, image, metadata in zip(text_samples, image_samples, metadata_samples):
            search_engine.add_content(text=text, image=image, metadata=metadata)
        
        # Temel arama - varsayılan parametrelerle
        query_text = "haber makale blog"
        default_results = search_engine.search(text=query_text, top_k=5)
        
        print("\nVarsayılan Arama Sonuçları:")
        for i, result in enumerate(default_results):
            print(f"{i+1}. {result['metadata']['category']} - Benzerlik: {result['score']:.4f}")
        
        # Özelleştirilmiş aramalar
        customization_scenarios = [
            {
                "name": "Yüksek Hassasiyet",
                "params": {"similarity_threshold": 0.8, "exact_match_boost": 0.1},
                "expectation": "Daha yüksek benzerlik eşiği daha az sonuç getirmeli"
            },
            {
                "name": "Kategori Ağırlıklandırma",
                "params": {"field_weights": {"category": 2.0}},
                "expectation": "Kategori alanına daha fazla ağırlık vermeli"
            },
            {
                "name": "Metin-Görüntü Dengesi",
                "params": {"text_image_ratio": 0.7},
                "expectation": "Metne görüntüden daha fazla ağırlık vermeli"
            },
            {
                "name": "Çeşitlilik Artırma",
                "params": {"diversity_factor": 0.8},
                "expectation": "Daha çeşitli kategorilerden sonuçlar getirmeli"
            }
        ]
        
        # Her özelleştirme senaryosunu test et
        for scenario in customization_scenarios:
            custom_results = search_engine.search(
                text=query_text, 
                top_k=5,
                **scenario["params"]
            )
            
            print(f"\n{scenario['name']} Arama Sonuçları:")
            for i, result in enumerate(custom_results):
                print(f"{i+1}. {result['metadata']['category']} - Benzerlik: {result['score']:.4f}")
            
            # Özelleştirme etkisini basitçe kontrol et - en azından bazı sonuçlar farklı olmalı
            default_ids = [r["metadata"]["id"] for r in default_results]
            custom_ids = [r["metadata"]["id"] for r in custom_results]
            
            has_differences = not (set(default_ids) == set(custom_ids) and 
                                 default_ids == custom_ids)  # Hem içerik hem sıralama kontrolü
            
            print(f"Sonuçlarda Fark Var mı: {has_differences}")
            print(f"Beklenti: {scenario['expectation']}")
            
            assert has_differences, \
                f"{scenario['name']} senaryosu varsayılan sonuçlardan farklı sonuçlar üretmedi"
    
    @pytest.mark.search
    def test_search_results_merging(self):
        """Arama sonuçları birleştirme testi"""
        _, search_engine, _ = self.setup_model_and_search()
        
        # Test verisi hazırla ve ekle
        text_samples, image_samples, metadata_samples = self.generate_test_data(count=40)
        
        for text, image, metadata in zip(text_samples, image_samples, metadata_samples):
            search_engine.add_content(text=text, image=image, metadata=metadata)
        
        # Farklı sorgular için sonuçlar al
        query1_results = search_engine.search(text="haber politika", top_k=5)
        query2_results = search_engine.search(text="ekonomi finans", top_k=5)
        query3_results = search_engine.search(text="teknoloji bilim", top_k=5)
        
        # Sonuçları farklı stratejilerle birleştir
        merge_strategies = [
            {
                "name": "Basit Birleştirme",
                "strategy": "simple",
                "weights": None,
                "expectation": "Tüm sonuçları birleştirmeli, tekrarları elemeli"
            },
            {
                "name": "Ağırlıklı Birleştirme",
                "strategy": "weighted",
                "weights": [0.5, 0.3, 0.2],  # İlk sorgu daha önemli
                "expectation": "İlk sorgudaki sonuçlar daha yüksek sıralanmalı"
            },
            {
                "name": "Sıralı Birleştirme",
                "strategy": "sequential",
                "weights": None,
                "expectation": "Önce ilk sorgu sonuçları, sonra diğerleri"
            },
            {
                "name": "Kesişim",
                "strategy": "intersection",
                "weights": None,
                "expectation": "Yalnızca tüm sorguların ortak sonuçları (çok az veya hiç olmayabilir)"
            }
        ]
        
        # Her stratejiyi test et
        for strategy_info in merge_strategies:
            # Sonuçları birleştir
            if strategy_info["strategy"] == "simple":
                merged_results = search_engine.merge_results(
                    [query1_results, query2_results, query3_results],
                    strategy="simple"
                )
            elif strategy_info["strategy"] == "weighted":
                merged_results = search_engine.merge_results(
                    [query1_results, query2_results, query3_results],
                    strategy="weighted",
                    weights=strategy_info["weights"]
                )
            elif strategy_info["strategy"] == "sequential":
                merged_results = search_engine.merge_results(
                    [query1_results, query2_results, query3_results],
                    strategy="sequential"
                )
            elif strategy_info["strategy"] == "intersection":
                merged_results = search_engine.merge_results(
                    [query1_results, query2_results, query3_results],
                    strategy="intersection"
                )
            
            print(f"\n{strategy_info['name']} Birleştirme Sonuçları:")
            print(f"Sonuç Sayısı: {len(merged_results)}")
            
            # İlk 3 sonucu yazdır
            for i, result in enumerate(merged_results[:3]):
                print(f"{i+1}. {result['metadata']['id']} - Benzerlik: {result['score']:.4f}")
            
            # Temel doğrulamalar
            if strategy_info["strategy"] == "simple":
                # Basit birleştirmede tekrarlar elenmeli
                result_ids = [r["metadata"]["id"] for r in merged_results]
                assert len(result_ids) == len(set(result_ids)), "Basit birleştirmede tekrarlanan ID'ler var"
                
            elif strategy_info["strategy"] == "weighted":
                # Ağırlıklı birleştirmede ilk sorgunun sonuçları daha yüksek sıralanmalı
                if len(merged_results) > 0 and len(query1_results) > 0:
                    highest_q1_id = query1_results[0]["metadata"]["id"]
                    merged_ids = [r["metadata"]["id"] for r in merged_results[:3]]  # İlk 3 sonuç
                    assert highest_q1_id in merged_ids, "Ağırlıklı birleştirmede ilk sorgunun en iyi sonucu üst sıralarda değil"
                
            elif strategy_info["strategy"] == "sequential":
                # Sıralı birleştirmede ilk sonuçlar ilk sorgudan gelmeli
                if len(merged_results) > 0 and len(query1_results) > 0:
                    first_merged = merged_results[0]["metadata"]["id"]
                    first_q1 = query1_results[0]["metadata"]["id"]
                    assert first_merged == first_q1, "Sıralı birleştirmede ilk sonuç ilk sorgudan gelmedi"
                
            elif strategy_info["strategy"] == "intersection":
                # Kesişim stratejisinde tüm sonuçlar her sorgu sonucunda da olmalı
                # (Gerçek verilerle kesişim boş olabilir, bu nedenle yumuşak bir test yapalım)
                if len(merged_results) > 0:
                    # Örnek olarak ilk sonucu kontrol edelim
                    intersect_id = merged_results[0]["metadata"]["id"]
                    in_query1 = any(r["metadata"]["id"] == intersect_id for r in query1_results)
                    in_query2 = any(r["metadata"]["id"] == intersect_id for r in query2_results)
                    in_query3 = any(r["metadata"]["id"] == intersect_id for r in query3_results)
                    
                    assert in_query1 and in_query2 and in_query3, \
                        "Kesişim stratejisinde sonuçlar tüm sorgu sonuçlarında bulunmalı"
    
    @pytest.mark.search
    def test_search_results_caching(self):
        """Arama sonuçları önbelleği testi"""
        _, search_engine, _ = self.setup_model_and_search()
        
        # Test verisi hazırla ve ekle
        text_samples, image_samples, metadata_samples = self.generate_test_data(count=30)
        
        for text, image, metadata in zip(text_samples, image_samples, metadata_samples):
            search_engine.add_content(text=text, image=image, metadata=metadata)
        
        # Önbellek etkinleştir (varsayılan olarak kapalıysa)
        search_engine.enable_cache(max_size=100, ttl_seconds=60)
        
        # Test sorguları
        test_queries = [
            "haber politika ekonomi",
            "teknoloji bilim araştırma",
            "sosyal medya",
            "ürün tanıtım"
        ]
        
        # Her sorgu için zamanla ve önbellek vuruşlarını kontrol et
        cache_hits = 0
        
        for query in test_queries:
            # İlk sorgu - önbellekte olmamalı
            start_time = time.time()
            first_results = search_engine.search(text=query, top_k=5)
            first_query_time = time.time() - start_time
            
            # İkinci sorgu - önbellekte olmalı
            start_time = time.time()
            second_results = search_engine.search(text=query, top_k=5)
            second_query_time = time.time() - start_time
            
            print(f"\nSorgu: '{query}'")
            print(f"İlk Sorgu Süresi: {first_query_time*1000:.2f} ms")
            print(f"İkinci Sorgu Süresi: {second_query_time*1000:.2f} ms")
            print(f"Hızlanma Oranı: {first_query_time/second_query_time:.2f}x")
            
            # İkinci sorgu daha hızlı olmalı (önbellekten geliyorsa)
            if second_query_time < first_query_time:
                cache_hits += 1
                print("Önbellek Vuruşu: EVET")
            else:
                print("Önbellek Vuruşu: HAYIR")
            
            # Sonuçlar aynı olmalı
            first_ids = [r["metadata"]["id"] for r in first_results]
            second_ids = [r["metadata"]["id"] for r in second_results]
            
            assert first_ids == second_ids, "Önbellekteki sonuçlar ilk sonuçlarla eşleşmiyor"
        
        # Önbellek vuruşları kontrolü
        print(f"\nToplam Önbellek Vuruşları: {cache_hits}/{len(test_queries)}")
        assert cache_hits > 0, "Hiç önbellek vuruşu gerçekleşmedi"
        
        # Önbellek temizleme
        search_engine.clear_cache()
        
        # Temizlik sonrası tekrar bir sorgu yapalım
        start_time = time.time()
        after_clear_results = search_engine.search(text=test_queries[0], top_k=5)
        after_clear_time = time.time() - start_time
        
        print(f"\nÖnbellek Temizleme Sonrası Sorgu Süresi: {after_clear_time*1000:.2f} ms")
        
        # Önbellek temizlendiği için süre ilk sorguya benzer olmalı
        assert len(after_clear_results) > 0, "Önbellek temizleme sonrası sonuç döndürülmedi" 
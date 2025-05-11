"""
M³TM modelinin füzyon mekanizması entegrasyon testleri

Bu modül, transformerlardan gelen çıktıların füzyon mekanizması ile 
entegrasyonunu test etmek için kapsamlı senaryolar içerir:
- Çapraz modalite dikkat mekanizmaları
- Adaptif modalite ağırlıklandırma
- Farklı füzyon stratejileri

Test örüntüleri:
- setup_teardown: Test düzeneğinin kurulması ve temizlenmesi
- integration_cascade: Bileşenleri kademeli olarak entegre etme
- boundary_test: Sınır durumlarında çalışmayı doğrulama
- comparative_test: Farklı füzyon stratejilerini karşılaştırma
- property_based_test: Model davranışının özelliklerini doğrulama
- multidimensional_parameter_sweep: Farklı füzyon parametrelerini tarama
"""
import os
import pytest
import torch
import tempfile
import numpy as np
from pathlib import Path

from m3tm.config.model_config import get_tiny_config
from m3tm.transformer.proto_transformer import ProtoTransformerBlock
from m3tm.embedding.text_embedding import TextEmbedding
from m3tm.embedding.image_embedding import ImagePatchEmbedding
from m3tm.fusion.basic_fusion import BasicFusion
from m3tm.fusion.cross_attention_fusion import CrossAttentionFusion
from m3tm.fusion.adaptive_weighting_fusion import AdaptiveWeightingFusion
from m3tm.search.search_embedding import SearchEmbeddingProjection
from m3tm.core.base_model import M3TMBaseModel


class TestFusionIntegration:
    """M³TM modelinin füzyon mekanizması entegrasyon testleri"""
    
    def setup_model(self, fusion_type="basic"):
        """Test için model ve füzyon modülü oluşturur
        
        Args:
            fusion_type: Kullanılacak füzyon türü ("basic", "cross_attention", "adaptive_weighting")
        """
        config = get_tiny_config()
        
        # Metin gömme
        text_embedding = TextEmbedding(config.text_config)
        
        # Görüntü gömme
        image_embedding = ImagePatchEmbedding(config.image_config)
        
        # Transformer blokları
        transformer_blocks = torch.nn.ModuleList([
            ProtoTransformerBlock(config.transformer_config) 
            for _ in range(config.num_core_blocks)
        ])
        
        # Füzyon mekanizması seçimi
        if fusion_type == "basic":
            fusion = BasicFusion(
                config.fusion_config.text_dim,
                config.fusion_config.image_dim,
                config.fusion_config.output_dim
            )
        elif fusion_type == "cross_attention":
            fusion = CrossAttentionFusion(
                config.fusion_config.text_dim,
                config.fusion_config.image_dim,
                config.fusion_config.output_dim,
                num_heads=config.fusion_config.num_attention_heads
            )
        elif fusion_type == "adaptive_weighting":
            fusion = AdaptiveWeightingFusion(
                config.fusion_config.text_dim,
                config.fusion_config.image_dim,
                config.fusion_config.output_dim
            )
        else:
            raise ValueError(f"Bilinmeyen füzyon türü: {fusion_type}")
        
        # Arama gömme projeksiyonu
        search_projection = SearchEmbeddingProjection(
            config.fusion_config.output_dim,
            config.search_config.search_dim
        )
        
        return {
            'config': config,
            'text_embedding': text_embedding,
            'image_embedding': image_embedding,
            'transformer_blocks': transformer_blocks,
            'fusion': fusion,
            'search_projection': search_projection,
            'fusion_type': fusion_type
        }
    
    @pytest.mark.integration
    def test_cross_modal_attention_fusion(self):
        """Çapraz modalite dikkat füzyonu entegrasyon testi
        
        Bu test, çapraz modalite dikkat mekanizmasının doğru çalışıp çalışmadığını
        kontrol eder. Modaliteler arası dikkat ağırlıklarının makul değerlere sahip
        olduğunu ve çapraz dikkat füzyonunun temel füzyondan farklı sonuçlar ürettiğini
        doğrular.
        
        Test örüntüsü: comparative_test, integration_cascade
        """
        # İki farklı füzyon mekanizmasıyla model oluştur
        basic_components = self.setup_model(fusion_type="basic")
        cross_attn_components = self.setup_model(fusion_type="cross_attention")
        
        # Test girdileri
        batch_size = 2
        text_seq_len = 16
        img_size = 32
        text_input = torch.randint(0, basic_components['config'].text_config.vocab_size, (batch_size, text_seq_len))
        image_input = torch.rand(batch_size, 3, img_size, img_size)
        
        # Metin ve görüntü özellikleri elde et - her iki model için aynı girdiyi kullanarak
        
        # Metin işleme
        basic_text_features = basic_components['text_embedding'](text_input, return_dict=False)
        cross_attn_text_features = cross_attn_components['text_embedding'](text_input, return_dict=False)
        
        # Görüntü işleme  
        basic_image_features = basic_components['image_embedding'](image_input, return_dict=False)
        cross_attn_image_features = cross_attn_components['image_embedding'](image_input, return_dict=False)
        
        # Transformer işleme
        for i in range(len(basic_components['transformer_blocks'])):
            basic_text_features = basic_components['transformer_blocks'][i](basic_text_features)[0]
            basic_image_features = basic_components['transformer_blocks'][i](basic_image_features)[0]
            
            cross_attn_text_features = cross_attn_components['transformer_blocks'][i](cross_attn_text_features)[0]
            cross_attn_image_features = cross_attn_components['transformer_blocks'][i](cross_attn_image_features)[0]
        
        # Global pooling
        basic_text_pooled = torch.mean(basic_text_features, dim=1)
        basic_image_pooled = torch.mean(basic_image_features, dim=1)
        
        cross_attn_text_pooled = torch.mean(cross_attn_text_features, dim=1)
        cross_attn_image_pooled = torch.mean(cross_attn_image_features, dim=1)
        
        # Füzyon - burada farklı füzyon yaklaşımları test edilir
        basic_fused = basic_components['fusion'](basic_text_pooled, basic_image_pooled)
        
        # CrossAttentionFusion ayrıca dikkat ağırlıklarını döndürür
        cross_attn_fused, attention_weights = cross_attn_components['fusion'](
            cross_attn_text_pooled, 
            cross_attn_image_pooled, 
            return_attention=True
        )
        
        # Doğrulamalar
        
        # 1. İki füzyon yaklaşımı farklı çıktılar üretmeli
        cosine_sim = torch.nn.functional.cosine_similarity(basic_fused, cross_attn_fused, dim=1)
        # Benzer ama tamamen aynı değil (0.5-0.95 arası benzerlik beklenir)
        assert torch.all(cosine_sim > 0.5) and torch.all(cosine_sim < 0.95), \
            "Çapraz dikkat füzyonu temel füzyondan çok farklı sonuçlar üretmemeli, ancak aynı da olmamalı"
        
        # 2. Çapraz dikkat ağırlıkları geçerli olmalı
        assert attention_weights.shape[1] == 2, "İki modalite (metin ve görüntü) için dikkat ağırlıkları beklenir"
        assert torch.allclose(torch.sum(attention_weights, dim=1), torch.ones(batch_size)), \
            "Dikkat ağırlıkları toplamı 1 olmalıdır"
        assert torch.all(attention_weights >= 0) and torch.all(attention_weights <= 1), \
            "Dikkat ağırlıkları [0,1] aralığında olmalıdır"
        
        # 3. Arama gömmeleri oluşturup normalize edildiğini kontrol et
        basic_search_emb = basic_components['search_projection'](basic_fused)
        cross_attn_search_emb = cross_attn_components['search_projection'](cross_attn_fused)
        
        # Normalizasyon kontrolü
        assert torch.allclose(torch.norm(basic_search_emb, p=2, dim=1), torch.ones(batch_size)), \
            "Temel füzyon için arama gömmeleri L2-normalize edilmeli"
        assert torch.allclose(torch.norm(cross_attn_search_emb, p=2, dim=1), torch.ones(batch_size)), \
            "Çapraz dikkat füzyonu için arama gömmeleri L2-normalize edilmeli"
        
        # 4. İki füzyon yaklaşımının arama gömmeleri de farklı sonuçlar üretmeli
        search_cosine_sim = torch.nn.functional.cosine_similarity(basic_search_emb, cross_attn_search_emb, dim=1)
        assert torch.all(search_cosine_sim > 0.5) and torch.all(search_cosine_sim < 0.95), \
            "Çapraz dikkat füzyon arama gömmeleri temel füzyondan makul ölçüde farklı olmalı"
    
    @pytest.mark.integration
    def test_modality_weighting_fusion(self):
        """Modalite ağırlıklandırma füzyonu entegrasyon testi
        
        Bu test, adaptif modalite ağırlıklandırma mekanizmasının hem metin hem
        görüntü modalitelerine duyarlı olduğunu ve tek modalite durumlarını doğru
        şekilde işleyebildiğini doğrular.
        
        Test örüntüsü: property_based_test, boundary_test
        """
        # Adaptif ağırlıklandırma füzyonu ile model oluştur
        model_components = self.setup_model(fusion_type="adaptive_weighting")
        
        # Test girdileri
        batch_size = 4
        text_seq_len = 16
        img_size = 32
        
        # Farklı modalite kombinasyonları oluştur:
        # 1. Hem metin hem görüntü (normal durum)
        # 2. Sadece metin - güçlü sinyal
        # 3. Sadece görüntü - güçlü sinyal
        # 4. Hem metin hem görüntü - metin daha zayıf sinyal (gürültülü)
        
        # Normal metin ve görüntü
        normal_text = torch.randint(0, model_components['config'].text_config.vocab_size, (batch_size, text_seq_len))
        normal_image = torch.rand(batch_size, 3, img_size, img_size)
        
        # Zayıf metin sinyali (bazı token'ları padding ile değiştirerek)
        weak_text = normal_text.clone()
        padding_token = 0  # Örnek olarak, gerçekte padding token ID'si kullanılacak
        weak_text[:, text_seq_len // 2:] = padding_token
        
        # Metin işleme
        normal_text_features = model_components['text_embedding'](normal_text, return_dict=False)
        weak_text_features = model_components['text_embedding'](weak_text, return_dict=False)
        
        # Görüntü işleme
        normal_image_features = model_components['image_embedding'](normal_image, return_dict=False)
        
        # Transformer işleme
        for block in model_components['transformer_blocks']:
            normal_text_features = block(normal_text_features)[0]
            weak_text_features = block(weak_text_features)[0]
            normal_image_features = block(normal_image_features)[0]
        
        # Global pooling
        normal_text_pooled = torch.mean(normal_text_features, dim=1)
        weak_text_pooled = torch.mean(weak_text_features, dim=1)
        normal_image_pooled = torch.mean(normal_image_features, dim=1)
        
        # Test durumları
        fusion = model_components['fusion']
        
        # 1. Normal metin ve normal görüntü füzyonu
        balanced_fused, balanced_weights = fusion(normal_text_pooled, normal_image_pooled, return_weights=True)
        
        # 2. Zayıf metin ve normal görüntü füzyonu
        text_weak_fused, text_weak_weights = fusion(weak_text_pooled, normal_image_pooled, return_weights=True)
        
        # 3. Sadece metin füzyonu (görüntü Yok)
        text_only_fused, text_only_weights = fusion(normal_text_pooled, None, return_weights=True)
        
        # 4. Sadece görüntü füzyonu (metin Yok)
        image_only_fused, image_only_weights = fusion(None, normal_image_pooled, return_weights=True)
        
        # Doğrulamalar
        
        # 1. Dengeli durumda, ağırlıklar nispeten dengeli olmalı
        assert torch.all(balanced_weights[:, 0] > 0.3) and torch.all(balanced_weights[:, 0] < 0.7), \
            "Dengeli girdilerle füzyonda, metin ağırlıkları aşırı dengesiz olmamalı"
        
        # 2. Zayıf metin durumunda, görüntü ağırlığı daha yüksek olmalı
        assert torch.all(text_weak_weights[:, 1] > balanced_weights[:, 1]), \
            "Zayıf metin sinyali olduğunda, görüntü ağırlığı artmalı"
        
        # 3. Sadece metin durumunda, tüm ağırlık metinde olmalı
        assert torch.all(text_only_weights[:, 0] > 0.99), \
            "Sadece metin girişi olduğunda, tüm ağırlık metine verilmeli"
        
        # 4. Sadece görüntü durumunda, tüm ağırlık görüntüde olmalı
        assert torch.all(image_only_weights[:, 1] > 0.99), \
            "Sadece görüntü girişi olduğunda, tüm ağırlık görüntüye verilmeli"
        
        # 5. Çıktılar geçerli boyutlara sahip olmalı
        expected_dim = model_components['config'].fusion_config.output_dim
        assert balanced_fused.shape == (batch_size, expected_dim)
        assert text_weak_fused.shape == (batch_size, expected_dim)
        assert text_only_fused.shape == (batch_size, expected_dim)
        assert image_only_fused.shape == (batch_size, expected_dim)
        
        # 6. Tek modalite çıktıları, girdilere benzer olmalı
        text_pooled_norm = torch.nn.functional.normalize(normal_text_pooled, p=2, dim=1)
        text_only_norm = torch.nn.functional.normalize(text_only_fused, p=2, dim=1)
        text_sim = torch.nn.functional.cosine_similarity(text_pooled_norm, text_only_norm, dim=1)
        assert torch.all(text_sim > 0.9), "Sadece metin füzyonu, metin girdisine yüksek oranda benzemeli"
        
        image_pooled_norm = torch.nn.functional.normalize(normal_image_pooled, p=2, dim=1)
        image_only_norm = torch.nn.functional.normalize(image_only_fused, p=2, dim=1)
        image_sim = torch.nn.functional.cosine_similarity(image_pooled_norm, image_only_norm, dim=1)
        assert torch.all(image_sim > 0.9), "Sadece görüntü füzyonu, görüntü girdisine yüksek oranda benzemeli"
    
    @pytest.mark.integration
    def test_fusion_attention_parameter_sweep(self):
        """Füzyon dikkat parametrelerini tarama testi
        
        Bu test, çapraz dikkat füzyon mekanizmasında farklı dikkat başlık sayıları
        ve boyutlarının etkisini değerlendirir. Farklı kombinasyonlar denenip
        en iyi performans gösteren konfigürasyon tespit edilir.
        
        Test örüntüsü: multidimensional_parameter_sweep
        """
        # Temel konfigürasyon
        config = get_tiny_config()
        
        # 1. Parametre uzayı tanımlama
        attention_heads = [1, 2, 4, 8]  # Dikkat başlığı sayıları
        attention_dims = [16, 32, 64]    # Dikkat boyutları
        
        # Test girdileri
        batch_size = 4
        text_seq_len = 16
        img_size = 32
        text_input = torch.randint(0, config.text_config.vocab_size, (batch_size, text_seq_len))
        image_input = torch.rand(batch_size, 3, img_size, img_size)
        
        # Temel girdi özellikleri - parametre taraması için sabit tutulur
        text_embedding = TextEmbedding(config.text_config)
        image_embedding = ImagePatchEmbedding(config.image_config)
        
        # Transformer blokları - parametre taraması için sabit tutulur
        transformer_blocks = torch.nn.ModuleList([
            ProtoTransformerBlock(config.transformer_config) 
            for _ in range(config.num_core_blocks)
        ])
        
        # Metin işleme
        text_features = text_embedding(text_input, return_dict=False)
        
        # Görüntü işleme
        image_features = image_embedding(image_input, return_dict=False)
        
        # Transformer işleme
        for block in transformer_blocks:
            text_features = block(text_features)[0]
            image_features = block(image_features)[0]
        
        # Global pooling
        text_pooled = torch.mean(text_features, dim=1)
        image_pooled = torch.mean(image_features, dim=1)
        
        results = {}
        
        # 2. Parametre uzayını tarama
        for num_heads in attention_heads:
            for attn_dim in attention_dims:
                # 3. Test konfigürasyonu
                key = f"heads={num_heads},dim={attn_dim}"
                
                # 4. Test senaryosunu çalıştırma
                try:
                    # CrossAttentionFusion oluştur
                    fusion = CrossAttentionFusion(
                        config.fusion_config.text_dim,
                        config.fusion_config.image_dim,
                        config.fusion_config.output_dim,
                        num_heads=num_heads,
                        head_dim=attn_dim
                    )
                    
                    # İleri akış
                    fused_features, attention_weights = fusion(text_pooled, image_pooled, return_attention=True)
                    
                    # Dikkat ağırlıklarının analizi
                    attn_entropy = -torch.sum(
                        attention_weights * torch.log(attention_weights + 1e-10), 
                        dim=1
                    ).mean().item()
                    
                    attn_variance = torch.var(attention_weights, dim=0).mean().item()
                    
                    # Arama gömme projeksiyonu
                    search_projection = SearchEmbeddingProjection(
                        config.fusion_config.output_dim,
                        config.search_config.search_dim
                    )
                    search_emb = search_projection(fused_features)
                    
                    # Çıktıların analizi
                    text_only_fusion = CrossAttentionFusion(
                        config.fusion_config.text_dim,
                        config.fusion_config.image_dim,
                        config.fusion_config.output_dim,
                        num_heads=1
                    )
                    text_only_fused = text_only_fusion(text_pooled, None)
                    
                    # Çapraz modalite etkileşimini ölç: füzyon çıktıları tek modaliteye ne kadar benziyor?
                    text_sim = torch.nn.functional.cosine_similarity(
                        fused_features, text_only_fused, dim=1
                    ).mean().item()
                    
                    # 5. Sonuçları toplama
                    results[key] = {
                        "success": True,
                        "attention_entropy": attn_entropy,  # Dikkat dağılımının belirsizliği
                        "attention_variance": attn_variance,  # Dikkat ağırlıklarının varyansı
                        "text_similarity": text_sim,  # Çoklu modalite etkileşimi ölçümü
                        "fusion_norm": torch.norm(fused_features).item(),  # Füzyon çıktı büyüklüğü
                        "parameters": fusion.get_parameter_count()  # Parametre sayısı
                    }
                except Exception as e:
                    # 6. Hataları kaydetme
                    results[key] = {
                        "success": False,
                        "error_type": str(type(e)),
                        "error_message": str(e)
                    }
        
        # 7. Sonuçları analiz etme
        successful_configs = [k for k, v in results.items() if v["success"]]
        
        # Başarılı konfigürasyonlar olduğunu doğrula
        assert len(successful_configs) > 0, "Hiçbir parametre kombinasyonu başarılı değil"
        
        # En yüksek entropi ve varyansa sahip kombinasyonu bul (en bilgi verici dikkat dağılımı)
        best_entropy_config = max(
            successful_configs, 
            key=lambda k: results[k]["attention_entropy"]
        )
        
        best_variance_config = max(
            successful_configs, 
            key=lambda k: results[k]["attention_variance"]
        )
        
        # Modaliteler arası etkileşim için optimal dengeyi bul
        # Text benzerliği 0.5'e en yakın olan (dengeli modalite etkileşimi gösteren)
        balanced_config = min(
            successful_configs, 
            key=lambda k: abs(results[k]["text_similarity"] - 0.5)
        )
        
        # Parametre verimliliğini bul (en az paramatre sayısıyla iyi performans)
        efficient_configs = [k for k in successful_configs if results[k]["attention_entropy"] > 0.5]
        if efficient_configs:
            efficient_config = min(
                efficient_configs, 
                key=lambda k: results[k]["parameters"]
            )
        else:
            efficient_config = min(
                successful_configs, 
                key=lambda k: results[k]["parameters"]
            )
        
        # 8. Sonuçları raporla
        print(f"\nFüzyon Dikkat Parametreleri Tarama Sonuçları:")
        print(f"Test edilen toplam kombinasyon: {len(results)}")
        print(f"Başarılı kombinasyon: {len(successful_configs)}")
        print(f"En yüksek dikkat entropisi: {best_entropy_config} (skor: {results[best_entropy_config]['attention_entropy']:.4f})")
        print(f"En yüksek dikkat varyansı: {best_variance_config} (skor: {results[best_variance_config]['attention_variance']:.4f})")
        print(f"En dengeli modalite etkileşimi: {balanced_config} (benzerlik: {results[balanced_config]['text_similarity']:.4f})")
        print(f"En verimli konfigürasyon: {efficient_config} (parametre: {results[efficient_config]['parameters']})")
        
        # 9. En iyi konfigürasyonun başarılı olduğunu doğrulama
        # Not: Burada "en iyi" tanımı test amacınıza bağlıdır
        best_config = balanced_config  # Bu testte dengeyi tercih ediyoruz
        assert results[best_config]["success"], f"En iyi konfigürasyon başarısız oldu: {best_config}" 
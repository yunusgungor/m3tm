"""
M³TM modeli için temel sınıf.
"""

import os
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from m3tm.config.model_config import M3TMConfig


class BaseModel(nn.Module):
    """Tüm M³TM modellerinin temel sınıfı."""
    
    def __init__(self, config: M3TMConfig):
        """
        BaseModel sınıfını başlatır.
        
        Args:
            config: Model yapılandırması
        """
        super().__init__()
        self.config = config
    
    def save_pretrained(self, save_dir: str) -> None:
        """
        Modeli ve yapılandırmasını verilen dizine kaydeder.
        
        Args:
            save_dir: Kaydetme dizini
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Model durumunu kaydet
        model_path = os.path.join(save_dir, "model.pt")
        torch.save(self.state_dict(), model_path)
        
        # Yapılandırmayı kaydet
        config_path = os.path.join(save_dir, "config.pt")
        torch.save(self.config, config_path)
    
    @classmethod
    def from_pretrained(cls, load_dir: str) -> "BaseModel":
        """
        Önceden eğitilmiş modeli ve yapılandırmasını verilen dizinden yükler.
        
        Args:
            load_dir: Yükleme dizini
            
        Returns:
            Yüklenmiş BaseModel örneği
        """
        # Yapılandırmayı yükle
        config_path = os.path.join(load_dir, "config.pt")
        config = torch.load(config_path)
        
        # Modeli oluştur
        model = cls(config)
        
        # Model durumunu yükle
        model_path = os.path.join(load_dir, "model.pt")
        model.load_state_dict(torch.load(model_path))
        
        return model
    
    def get_param_count(self) -> Dict[str, int]:
        """
        Modeldeki parametrelerin sayısını hesaplar.
        
        Returns:
            Parametre istatistikleri içeren sözlük:
            - total: Toplam parametre sayısı
            - trainable: Eğitilebilir parametre sayısı
            - frozen: Donmuş parametre sayısı
        """
        total_params = 0
        trainable_params = 0
        
        for param in self.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": total_params - trainable_params
        }
    
    def forward(self, *args, **kwargs):
        """
        İleri beslemeli geçiş.
        Alt sınıflarda uygulanmalıdır.
        """
        raise NotImplementedError("BaseModel.forward alt sınıflarda uygulanmalıdır.") 


class M3TMBaseModel(BaseModel):
    """
    Multimodal Mobile Model Transformer (M³TM) temel modeli.
    
    Bu sınıf, BaseModel'i genişleterek M³TM mimarisinin temel bileşenlerini içerir:
    - Metin gömme (text_embedding)
    - Görüntü yama gömme (image_embedding)
    - Transformer blokları (transformer_blocks)
    - Füzyon (fusion)
    - Arama gömme projeksiyonu (search_embedding)
    """
    
    def __init__(self, config: M3TMConfig):
        """
        M3TMBaseModel sınıfını başlatır.
        
        Args:
            config: Model yapılandırması
        """
        super().__init__(config)
        
        # Metin gömme
        self.text_embedding = None
        if config.use_text_modality:
            from m3tm.embedding.text_embedding import TextEmbedding
            self.text_embedding = TextEmbedding(config.text_config)
        
        # Görüntü yama gömme
        self.image_embedding = None
        if config.use_image_modality:
            from m3tm.embedding.image_embedding import ImagePatchEmbedding
            self.image_embedding = ImagePatchEmbedding(config.image_config)
        
        # Transformer blokları
        from m3tm.transformer.proto_transformer import ProtoTransformerBlock
        self.transformer_blocks = nn.ModuleList([
            ProtoTransformerBlock(config.transformer_config) 
            for _ in range(config.num_transformer_blocks)
        ])
        
        # Füzyon
        from m3tm.fusion.basic_fusion import BasicFusion
        from m3tm.fusion.config import FusionConfig as FusionModuleConfig, FusionType

        # Model config'deki FusionConfig'i fusion modülündeki FusionConfig'e dönüştür
        fusion_config = FusionModuleConfig(
            fusion_type=FusionType.CONCATENATION,  # Varsayılan olarak concatenation
            text_dim=config.fusion_config.text_dim,
            image_dim=config.fusion_config.image_dim,
            output_dim=config.fusion_config.output_dim,
            use_layer_norm=config.fusion_config.use_layer_norm,
            dropout_rate=config.fusion_config.dropout
        )

        self.fusion = BasicFusion(fusion_config)
        
        # Arama gömme projeksiyonu
        from m3tm.search.search_embedding import SearchEmbeddingProjection
        self.search_embedding = SearchEmbeddingProjection(config.search_config)

        # Classification head (eğitim için)
        self.classifier = nn.Linear(config.fusion_config.output_dim, 2)  # Binary classification için
    
    def forward(self, text_input=None, image_input=None, return_dict=True):
        """
        İleri beslemeli geçiş.
        
        Args:
            text_input: Metin girişi (opsiyonel)
            image_input: Görüntü girişi (opsiyonel)
            return_dict: Sözlük olarak dön (varsayılan: True)
            
        Returns:
            Çıktı değerleri, return_dict=True ise sözlük olarak
        """
        outputs = {}
        
        # Metin gömme
        text_features = None
        if text_input is not None and self.text_embedding is not None:
            # text_input bir dict ise input_ids'i çıkar
            if isinstance(text_input, dict):
                input_ids = text_input.get('input_ids')
                attention_mask = text_input.get('attention_mask')
                text_embedding_output = self.text_embedding(input_ids, attention_mask)
            else:
                # text_input doğrudan tensor ise
                text_embedding_output = self.text_embedding(text_input)

            # TextEmbedding dict döndürüyorsa embeddings'i çıkar
            if isinstance(text_embedding_output, dict):
                text_features = text_embedding_output.get('embeddings')
            else:
                text_features = text_embedding_output

            outputs["text_features"] = text_features
        
        # Görüntü gömme
        image_features = None
        if image_input is not None and self.image_embedding is not None:
            image_features = self.image_embedding(image_input)
            outputs["image_features"] = image_features
        
        # Transformer blokları - text_features için
        if text_features is not None:
            # text_features sözlük olabilir - doğrudan tensor almalıyız
            text_tensor = text_features
            if isinstance(text_tensor, dict):
                if 'hidden_states' in text_tensor:
                    text_tensor = text_tensor['hidden_states']
                elif 'embeddings' in text_tensor:
                    text_tensor = text_tensor['embeddings']
                elif len(text_tensor) == 1:
                    text_tensor = list(text_tensor.values())[0]
                    
            # Artık text_tensor gerçekten bir tensor
            for transformer_block in self.transformer_blocks:
                transformer_output = transformer_block(text_tensor)
                
                # transformer_block tuple döndürür (output, metrics)
                if isinstance(transformer_output, tuple) and len(transformer_output) >= 1:
                    text_tensor = transformer_output[0]
                    # Eğer transformer_output[0] bir sözlükse, içerisindeki tensör değerini alalım
                    if isinstance(text_tensor, dict):
                        if 'hidden_states' in text_tensor:
                            text_tensor = text_tensor['hidden_states']
                        elif 'embeddings' in text_tensor:
                            text_tensor = text_tensor['embeddings']
                        elif len(text_tensor) == 1:
                            text_tensor = list(text_tensor.values())[0]
                else:
                    text_tensor = transformer_output
                    # Eğer text_tensor bir sözlükse, içerisindeki tensör değerini alalım
                    if isinstance(text_tensor, dict):
                        if 'hidden_states' in text_tensor:
                            text_tensor = text_tensor['hidden_states']
                        elif 'embeddings' in text_tensor:
                            text_tensor = text_tensor['embeddings']
                        elif len(text_tensor) == 1:
                            text_tensor = list(text_tensor.values())[0]
                
            # Çıktı tensörünü kaydet
            outputs["transformed_text"] = text_tensor
            text_features = text_tensor  # fusion için güncelle
        
        # Transformer blokları - image_features için
        if image_features is not None:
            # image_features sözlük olabilir - doğrudan tensor almalıyız
            image_tensor = image_features
            if isinstance(image_tensor, dict):
                if 'hidden_states' in image_tensor:
                    image_tensor = image_tensor['hidden_states']
                elif 'embeddings' in image_tensor:
                    image_tensor = image_tensor['embeddings']
                elif len(image_tensor) == 1:
                    image_tensor = list(image_tensor.values())[0]
                    
            # Artık image_tensor gerçekten bir tensor
            for transformer_block in self.transformer_blocks:
                transformer_output = transformer_block(image_tensor)
                
                # transformer_block tuple döndürür (output, metrics)
                if isinstance(transformer_output, tuple) and len(transformer_output) >= 1:
                    image_tensor = transformer_output[0]
                    # Eğer transformer_output[0] bir sözlükse, içerisindeki tensör değerini alalım
                    if isinstance(image_tensor, dict):
                        if 'hidden_states' in image_tensor:
                            image_tensor = image_tensor['hidden_states']
                        elif 'embeddings' in image_tensor:
                            image_tensor = image_tensor['embeddings']
                        elif len(image_tensor) == 1:
                            image_tensor = list(image_tensor.values())[0]
                else:
                    image_tensor = transformer_output
                    # Eğer image_tensor bir sözlükse, içerisindeki tensör değerini alalım
                    if isinstance(image_tensor, dict):
                        if 'hidden_states' in image_tensor:
                            image_tensor = image_tensor['hidden_states']
                        elif 'embeddings' in image_tensor:
                            image_tensor = image_tensor['embeddings']
                        elif len(image_tensor) == 1:
                            image_tensor = list(image_tensor.values())[0]
                
            # Çıktı tensörünü kaydet
            outputs["transformed_image"] = image_tensor
            image_features = image_tensor  # fusion için güncelle
        
        # Füzyon
        fused_features = None
        if text_features is not None or image_features is not None:
            # text_features ve image_features artık gerçek tensörler olmalı
            # BasicFusion modülü [batch_size, feature_dim] şeklinde tensörler bekliyor,
            # ancak Transformer çıktıları [batch_size, seq_len, feature_dim] şeklinde.
            # Bu nedenle, seq_len ekseni boyunca ortalama alarak boyutları uyumlu hale getiriyoruz.
            
            if text_features is not None:
                # Tensor şekli kontrol et ve gerekirse ortalama al
                if len(text_features.shape) == 3:  # [batch_size, seq_len, feature_dim]
                    text_features = torch.mean(text_features, dim=1)  # [batch_size, feature_dim]
            
            if image_features is not None:
                # Tensor şekli kontrol et ve gerekirse ortalama al
                if len(image_features.shape) == 3:  # [batch_size, seq_len, feature_dim]
                    image_features = torch.mean(image_features, dim=1)  # [batch_size, feature_dim]
            
            fused_features = self.fusion(text_features, image_features, return_dict=False)
            outputs["fused_features"] = fused_features
        
        # Arama gömme
        if fused_features is not None:
            search_embedding_output = self.search_embedding(fused_features)

            # search_embedding_output bir tensor veya dict olabilir
            if isinstance(search_embedding_output, dict):
                # search_embedding_output bir dict - içerden projections'ı al
                search_embedding = search_embedding_output.get("projections", search_embedding_output)
                # Tam çıktıyı da saklayalım
                outputs.update(search_embedding_output)
                outputs["search_embedding"] = search_embedding
            else:
                # search_embedding_output direkt bir tensor
                search_embedding = search_embedding_output
                outputs["search_embedding"] = search_embedding

            # Classification logits
            logits = self.classifier(fused_features)
            outputs["logits"] = logits

        if not return_dict:
            return outputs.get("search_embedding", None)

        return outputs
    
    def add_adapter(self, adapter_name: str, layer_id: int = 0):
        """
        Belirtilen katmana bir adapter ekler.
        
        Args:
            adapter_name: Adapter adı
            layer_id: Adapter eklenecek katman indeksi (varsayılan: 0)
        """
        if 0 <= layer_id < len(self.transformer_blocks):
            from m3tm.adapters.adapter_manager import create_adapter
            adapter = create_adapter(
                input_dim=self.config.transformer_config.embed_dim,
                bottleneck_dim=self.config.adapter_config.bottleneck_dim
            )
            self.transformer_blocks[layer_id].add_adapter(adapter_name, adapter)
    
    def get_search_embedding(self, text_input=None, image_input=None):
        """
        Verilen girdiler için arama gömme vektörünü döndürür.
        
        Args:
            text_input: Metin girişi (opsiyonel)
            image_input: Görüntü girişi (opsiyonel)
            
        Returns:
            Arama gömme vektörü
        """
        outputs = self.forward(text_input=text_input, image_input=image_input)
        return outputs.get("search_embedding", None) 
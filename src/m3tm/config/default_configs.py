"""
Varsayılan M³TM Yapılandırmaları

Bu modül, farklı senaryolar için önceden tanımlanmış yapılandırmalar içerir.
Farklı boyutlarda modeller ve çeşitli kullanım durumları için hazır yapılandırmalar sağlar.
"""

from .model_config import (
    M3TMModelConfig, 
    TextEmbeddingConfig, 
    ImagePatchEmbeddingConfig,
    TransformerConfig,
    FusionConfig,
    SearchConfig,
    AdapterConfig,
    TaskHeadConfig,
    TrainingConfig
)


def create_tiny_config() -> M3TMModelConfig:
    """
    Çok küçük boyutlu bir model yapılandırması oluşturur.
    
    Bu yapılandırma, test ve hızlı prototipleme için uygundur.
    Çok az parametre içerir ve minimum sistem gereksinimleri ile çalışabilir.
    
    Returns:
        M3TMModelConfig: Tiny model yapılandırması
    """
    config = M3TMModelConfig(
        name="m3tm-v2.3-tiny",
        description="M³TM Tiny - Temel test ve prototipleme için minimum boyutlu model",
        version="2.3.0",
        
        text_config=TextEmbeddingConfig(
            vocab_size=1000,
            embed_dim=32,
            max_seq_len=128
        ),
        
        image_config=ImagePatchEmbeddingConfig(
            embed_dim=32,
            patch_size=8,
            image_size=(112, 112)
        ),
        
        transformer_config=TransformerConfig(
            embed_dim=32,
            num_heads=1,
            mlp_ratio=2.0,
            dropout=0.1,
            attention_dropout=0.1
        ),
        
        fusion_config=FusionConfig(
            text_dim=32,
            image_dim=32,
            output_dim=64,
            fusion_type="concat"
        ),
        
        search_config=SearchConfig(
            input_dim=64,
            search_dim=64,
            index_type="simple"
        ),
        
        adapter_config=AdapterConfig(
            embed_dim=32,
            reduction_factor=8
        ),
        
        training_config=TrainingConfig(
            batch_size=4,
            learning_rate=0.001,
            epochs=5
        ),
        
        num_transformer_blocks=1
    )
    
    return config


def create_small_config() -> M3TMModelConfig:
    """
    Küçük boyutlu bir model yapılandırması oluşturur.
    
    Düşük kaynaklı mobil cihazlarda kullanım için uygundur.
    Makul performans ve küçük model boyutu sunar.
    
    Returns:
        M3TMModelConfig: Small model yapılandırması
    """
    config = M3TMModelConfig(
        name="m3tm-v2.3-small",
        description="M³TM Small - Düşük kaynaklı mobil cihazlar için küçük model",
        version="2.3.0",
        
        text_config=TextEmbeddingConfig(
            vocab_size=4000,
            embed_dim=64,
            max_seq_len=256
        ),
        
        image_config=ImagePatchEmbeddingConfig(
            embed_dim=64,
            patch_size=6,
            image_size=(192, 192)
        ),
        
        transformer_config=TransformerConfig(
            embed_dim=64,
            num_heads=2,
            mlp_ratio=2.5,
            attention_type="MobileAttention"
        ),
        
        fusion_config=FusionConfig(
            text_dim=64,
            image_dim=64,
            output_dim=128,
            fusion_type="concat"
        ),
        
        search_config=SearchConfig(
            input_dim=128,
            search_dim=128,
            index_type="faiss"
        ),
        
        adapter_config=AdapterConfig(
            embed_dim=64,
            reduction_factor=16
        ),
        
        training_config=TrainingConfig(
            batch_size=8,
            learning_rate=0.001
        ),
        
        num_transformer_blocks=2
    )
    
    return config


def create_base_config() -> M3TMModelConfig:
    """
    Temel boyutlu model yapılandırması oluşturur.
    
    Bu, çoğu kullanım durumu için iyi bir denge sunan varsayılan yapılandırmadır.
    Makul model boyutu ve iyi performans sağlar.
    
    Returns:
        M3TMModelConfig: Base model yapılandırması
    """
    # M3TMModelConfig zaten varsayılan olarak base config değerlerini içeriyor
    # Burada getDefault kullanarak alıp bazı değerleri özelleştirebiliriz
    return M3TMModelConfig.get_default_config()


def create_text_only_config() -> M3TMModelConfig:
    """
    Sadece metin modalitesi kullanan bir model yapılandırması oluşturur.
    
    Görüntü verisi olmayan veya gerektirmeyen uygulamalar için uygundur.
    
    Returns:
        M3TMModelConfig: Sadece metin yapılandırması
    """
    config = create_base_config()
    config.name = "m3tm-v2.3-text-only"
    config.description = "M³TM Text-Only - Sadece metin modalitesi kullanan model"
    
    # Görüntü modalitesini devre dışı bırak
    config.use_image_modality = False
    
    # Füzyon yapılandırmasını güncelle (sadece metin)
    config.fusion_config.fusion_type = "add"  # Tek modalite için daha basit
    config.fusion_config.output_dim = 64  # Doğrudan metin boyutu
    
    # Arama yapılandırmasını güncelle
    config.search_config.input_dim = 64
    
    return config


def create_image_only_config() -> M3TMModelConfig:
    """
    Sadece görüntü modalitesi kullanan bir model yapılandırması oluşturur.
    
    Metin verisi olmayan veya gerektirmeyen uygulamalar için uygundur.
    
    Returns:
        M3TMModelConfig: Sadece görüntü yapılandırması
    """
    config = create_base_config()
    config.name = "m3tm-v2.3-image-only"
    config.description = "M³TM Image-Only - Sadece görüntü modalitesi kullanan model"
    
    # Metin modalitesini devre dışı bırak
    config.use_text_modality = False
    
    # Görüntü işlemeyi güçlendir
    config.image_config.patch_size = 4  # Daha küçük yamalar
    
    # Füzyon yapılandırmasını güncelle (sadece görüntü)
    config.fusion_config.fusion_type = "add"  # Tek modalite için daha basit
    config.fusion_config.output_dim = 64  # Doğrudan görüntü boyutu
    
    # Arama yapılandırmasını güncelle
    config.search_config.input_dim = 64
    
    return config


def create_low_memory_config() -> M3TMModelConfig:
    """
    Düşük bellek kullanımına optimize edilmiş bir model yapılandırması oluşturur.
    
    Çok sınırlı belleğe sahip eski cihazlar için uygundur.
    
    Returns:
        M3TMModelConfig: Düşük bellek yapılandırması
    """
    config = create_small_config()  # Small'dan başla ve daha da optimize et
    config.name = "m3tm-v2.3-low-memory"
    config.description = "M³TM Low-Memory - Minimum bellek kullanımı için optimize edilmiş model"
    
    # Daha da düşük boyutlar kullan
    config.text_config.embed_dim = 32
    config.text_config.max_seq_len = 128
    
    config.image_config.embed_dim = 32
    config.image_config.patch_size = 8
    config.image_config.image_size = (112, 112)
    
    config.transformer_config.embed_dim = 32
    config.transformer_config.num_heads = 1
    config.transformer_config.mlp_ratio = 1.5  # Daha az genişleme
    
    config.fusion_config.text_dim = 32
    config.fusion_config.image_dim = 32
    config.fusion_config.output_dim = 48
    
    config.search_config.input_dim = 48
    config.search_config.search_dim = 48
    
    config.adapter_config.embed_dim = 32
    config.adapter_config.reduction_factor = 4
    
    # Transformer blok sayısını azalt
    config.num_transformer_blocks = 1
    
    return config


def create_high_performance_config() -> M3TMModelConfig:
    """
    Yüksek performansa optimize edilmiş bir model yapılandırması oluşturur.
    
    Üst düzey mobil cihazlar için uygundur.
    
    Returns:
        M3TMModelConfig: Yüksek performans yapılandırması
    """
    config = create_base_config()
    config.name = "m3tm-v2.3-high-performance"
    config.description = "M³TM High-Performance - Maksimum performans için optimize edilmiş model"
    
    # Daha geniş ve derin model
    config.text_config.embed_dim = 128
    config.text_config.max_seq_len = 512
    
    config.image_config.embed_dim = 128
    config.image_config.patch_size = 4
    config.image_config.image_size = (224, 224)
    
    config.transformer_config.embed_dim = 128
    config.transformer_config.num_heads = 4
    config.transformer_config.mlp_ratio = 4.0
    
    config.fusion_config.text_dim = 128
    config.fusion_config.image_dim = 128
    config.fusion_config.output_dim = 256
    config.fusion_config.fusion_type = "cross_attention"
    
    config.search_config.input_dim = 256
    config.search_config.search_dim = 256
    
    config.adapter_config.embed_dim = 128
    config.adapter_config.reduction_factor = 8
    
    # Daha fazla transformer blok
    config.num_transformer_blocks = 6
    
    return config


def create_classification_config() -> M3TMModelConfig:
    """
    Sınıflandırma görevleri için özelleştirilmiş bir model yapılandırması oluşturur.
    
    Metin ve görüntü sınıflandırması gibi görevler için uygundur.
    
    Returns:
        M3TMModelConfig: Sınıflandırma yapılandırması
    """
    config = create_base_config()
    config.name = "m3tm-v2.3-classification"
    config.description = "M³TM Classification - Sınıflandırma görevleri için optimize edilmiş model"
    
    # Daha geniş füzyon katmanı ve arama boyutu
    config.fusion_config.output_dim = 192
    config.search_config.input_dim = 192
    config.search_config.search_dim = 192
    
    # Sınıflandırma başlığı ekle (modelConfig'de direkt olmadığı için not olarak belirtelim)
    # Not: Gerçek uygulamada, bu yapılandırma ile bir TaskHead oluşturulabilir
    # task_head_config = TaskHeadConfig(
    #     task_type="classification",
    #     input_dim=192,
    #     hidden_dim=96,
    #     output_dim=10  # Örn: 10 sınıf
    # )
    
    return config


# Tüm yapılandırmaları içeren bir sözlük
PREDEFINED_CONFIGS = {
    "tiny": create_tiny_config,
    "small": create_small_config,
    "base": create_base_config,
    "text_only": create_text_only_config,
    "image_only": create_image_only_config,
    "low_memory": create_low_memory_config,
    "high_performance": create_high_performance_config,
    "classification": create_classification_config
}


def get_predefined_config(config_name: str) -> M3TMModelConfig:
    """
    Önceden tanımlanmış bir yapılandırmayı isimle alır.
    
    Args:
        config_name: Yapılandırma adı ('tiny', 'small', 'base', vb.)
        
    Returns:
        M3TMModelConfig: İstenilen yapılandırma
        
    Raises:
        ValueError: Geçersiz yapılandırma adı belirtilirse
    """
    if config_name not in PREDEFINED_CONFIGS:
        valid_configs = ", ".join(PREDEFINED_CONFIGS.keys())
        raise ValueError(
            f"Invalid config name: '{config_name}'. Valid options are: {valid_configs}"
        )
    
    # İlgili fabrika fonksiyonunu çağır
    return PREDEFINED_CONFIGS[config_name]() 
# Desen Kullanım Isı Haritası

```mermaid
heatmap
  title Modül-Desen Kullanım Yoğunluğu
  x-axis ["ConfigurationDataclass", "FactoryMethod", "ModelComposite", "DatasetFactory", "MetricsCollector", "ConfigValidationPipeline", "DataPreprocessingPipeline", "ConfigurationComposite", "TrainingLoopTemplate"]
  y-axis ["m3tm/config", "m3tm/embedding", "m3tm/transformer", "m3tm/task_heads", "m3tm/training", "m3tm/examples"]
  "m3tm/config"+"ConfigurationDataclass" 3
  "m3tm/config"+"ConfigValidationPipeline" 3
  "m3tm/config"+"ConfigurationComposite" 1
  
  "m3tm/embedding"+"ConfigurationDataclass" 1
  "m3tm/embedding"+"FactoryMethod" 1
  
  "m3tm/transformer"+"ConfigurationDataclass" 1
  
  "m3tm/task_heads"+"ConfigurationDataclass" 1
  "m3tm/task_heads"+"FactoryMethod" 1
  "m3tm/task_heads"+"ConfigValidationPipeline" 1
  "m3tm/task_heads"+"ConfigurationComposite" 1
  
  "m3tm/training"+"MetricsCollector" 3
  "m3tm/training"+"DatasetFactory" 3
  "m3tm/training"+"DataPreprocessingPipeline" 2
  "m3tm/training"+"TrainingLoopTemplate" 3
  "m3tm/training"+"ModelComposite" 2
  
  "m3tm/examples"+"ModelComposite" 1
  "m3tm/examples"+"TrainingLoopTemplate" 1
  "m3tm/examples"+"MetricsCollector" 1
```

## Gözlemler

- `m3tm/training` modülü en çok örüntü uygulanan modüldür
- `ConfigurationDataclass` en yaygın kullanılan örüntüdür, tüm yapılandırma dosyalarında kullanılmaktadır
- `ModelComposite` ve `TrainingLoopTemplate` örüntüleri yüksek etkileşim göstermektedir
- `m3tm/task_heads` modülü yapılandırma ve fabrika örüntülerinin odak noktasıdır

## Geliştirme Fırsatları

- `m3tm/examples` modülüne daha fazla desen uygulanabilir
- `m3tm/transformer` modülü için ek desenler değerlendirilebilir
- `DataPreprocessingPipeline` örüntüsü diğer modüllere de genişletilebilir 
# Adapter ve Task Head Eğitim Mekanizması Pattern Mimarisi

Bu diyagram, M³TM modelinde adapter ve task head eğitimi için kullanılan pattern'lerin ilişkilerini göstermektedir.

## Pattern İlişki Diyagramı

```mermaid
graph TD
    %% Ana pattern'ler
    FB[PT-018: FrozenBackboneTraining] --> CB[PT-019: TrainingCallbackHook]
    FB --> MO[PT-020: MemoryOptimization]
    FB --> CA[PT-014: CompositeAdapter]
    
    %% İlişkili pattern'ler
    CB --> TL[PT-013: TrainingLoopTemplate]
    CA --> MC[PT-003: ModelComposite]
    TL --> MC
    
    %% Anti-pattern'ler ve ilişkileri
    AP1[AP-001: MonolithicTraining] -.Çözümü.-> FB
    AP2[AP-002: CallbackHell] -.Çözümü.-> CB
    AP3[AP-003: DeepAdapterStack] -.Çözümü.-> CA
    
    %% Stil tanımlamaları
    classDef pattern fill:#a8d5ba,stroke:#178344,stroke-width:2px;
    classDef antipattern fill:#f8cecc,stroke:#b85450,stroke-width:2px;
    classDef related fill:#dae8fc,stroke:#6c8ebf,stroke-width:1px;
    
    %% Sınıfları uygula
    class FB,CB,MO pattern;
    class AP1,AP2,AP3 antipattern;
    class CA,TL,MC related;
```

## Adapter Eğitim Akış Diyagramı

```mermaid
sequenceDiagram
    participant User
    participant ATM as AdapterTrainingManager
    participant Model
    participant AM as AdapterManager
    participant TH as TaskHeads
    participant CA as Callbacks
    
    User->>ATM: Oluştur(model, config, adapter_manager)
    ATM->>AM: Adapter'ları al
    ATM->>Model: Görev başlıklarını bul
    
    User->>ATM: prepare_model_for_training()
    ATM->>Model: Çekirdek modeli dondur
    ATM->>AM: Adapter'ları eğitilebilir yap
    ATM->>TH: Görev başlıklarını eğitilebilir yap
    
    User->>ATM: train(train_loader, criterion)
    ATM->>CA: on_training_start() çağır
    
    loop Her epoch için
        ATM->>CA: on_epoch_start() çağır
        
        loop Her batch için
            ATM->>CA: on_batch_start() çağır
            ATM->>Model: İleri geçiş (adapter'larla)
            ATM->>Model: Geri yayılım (sadece adapter'lar ve başlıklar)
            ATM->>CA: on_batch_end() çağır
        end
        
        ATM->>Model: Değerlendir (val_loader)
        ATM->>ATM: Checkpoint kaydet
        ATM->>CA: on_epoch_end() çağır
        
        alt Erken durdurma
            CA->>ATM: Eğitimi durdur
        end
    end
    
    ATM->>CA: on_training_end() çağır
    ATM->>User: Eğitim metrikleri döndür
```

## Bileşen Diyagramı

```mermaid
classDiagram
    class AdapterTrainingConfig {
        +List~str~ train_adapter_names
        +bool train_task_heads
        +bool freeze_core_model
        +int checkpoint_frequency
        +str memory_optimization_level
    }
    
    class TrainingCallback {
        +on_training_start()
        +on_epoch_start()
        +on_batch_start()
        +on_batch_end()
        +on_epoch_end()
        +on_training_end()
    }
    
    class EarlyStoppingCallback {
        +int patience
        +str monitor
        +str mode
        +on_epoch_end()
    }
    
    class AdapterTrainingManager {
        -model
        -config
        -adapter_manager
        -save_dir
        +prepare_model_for_training()
        +train()
        +evaluate()
        +get_trainable_parameter_count()
    }
    
    class AdapterManager {
        +register_adapter()
        +list_adapters()
        +set_adapters_trainable()
    }
    
    AdapterTrainingManager o-- AdapterTrainingConfig
    AdapterTrainingManager o-- AdapterManager
    AdapterTrainingManager *-- TrainingCallback
    
    TrainingCallback <|-- EarlyStoppingCallback
    TrainingCallback <|-- LearningRateSchedulerCallback
    
    AdapterTrainingConfig --|> TrainingConfig
```

## Bellek Optimizasyonu Seviye Diyagramı

```mermaid
graph LR
    %% Ana seviyeler
    Low[Düşük Optimizasyon]
    Mod[Orta Optimizasyon]
    Agg[Yüksek Optimizasyon]
    
    %% Optimizasyon teknikleri
    GC["Gradient Checkpoint"]
    FP16["16-bit hassasiyet"]
    MC["Model tamponu temizleme"]
    IB["Ara bellekleri temizleme"]
    
    %% İlişkiler
    Low --> GC
    
    Mod --> GC
    Mod --> MC
    
    Agg --> GC
    Agg --> MC
    Agg --> FP16
    Agg --> IB
    
    %% Stil tanımlamaları
    classDef low fill:#d5f5e3,stroke:#1e8449,stroke-width:1px;
    classDef mod fill:#fdebd0,stroke:#e67e22,stroke-width:1px;
    classDef agg fill:#f5b7b1,stroke:#c0392b,stroke-width:1px;
    classDef tech fill:#eaecee,stroke:#2c3e50,stroke-width:1px;
    
    %% Sınıfları uygula
    class Low low;
    class Mod mod;
    class Agg agg;
    class GC,FP16,MC,IB tech;
```

Bu diyagramlar, M³TM modelindeki adapter ve task head eğitim mekanizmasının mimarisini, akışını, bileşenlerini ve bellek optimizasyon seviyelerini göstermektedir. 
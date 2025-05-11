# M³TM Android SDK Desen Görselleştirmesi

Bu görselleştirme dosyası, M³TM v2.3 Android SDK'sında kullanılan tasarım desenlerini ve aralarındaki ilişkileri göstermektedir.

## Desen İlişki Diyagramı

```mermaid
graph TD
    %% Ana desenler
    PT100[PT-100: Singleton Pattern]
    PT101[PT-101: Factory Method]
    PT102[PT-102: Bridge Pattern]
    PT103[PT-103: Facade Pattern]
    PT104[PT-104: Callback Pattern]
    PT105[PT-105: Resource Management]
    PT106[PT-106: Error Handling]
    
    %% İlişkileri tanımla
    PT100 --> PT103[Facade kullanır]
    PT101 --> PT100[Singleton nesneleri oluşturabilir]
    PT102 --> PT106[Hata dönüşümü]
    PT103 --> PT102[Bridge ile native tarafa erişim]
    PT105 --> PT102[Native kaynakları yönetir]
    PT104 --> PT102[Native taraftan callback'ler alır]
    
    %% Anti-pattern ilişkileri
    AP015[AP-015: Failed Resource Cleanup]
    AP016[AP-016: Excessive JNI Calls]
    AP017[AP-017: Unsynchronized Singleton Access]
    
    AP015 -.-> PT105[Karşı desen]
    AP016 -.-> PT102[Karşı desen]
    AP017 -.-> PT100[Karşı desen]
    
    %% Sınıf ilişkileri
    classDef singleton fill:#f9f,stroke:#333,stroke-width:2px
    classDef factory fill:#ff9,stroke:#333,stroke-width:2px
    classDef bridge fill:#9cf,stroke:#333,stroke-width:2px
    classDef facade fill:#9f9,stroke:#333,stroke-width:2px
    classDef callback fill:#fcf,stroke:#333,stroke-width:2px
    classDef resource fill:#cff,stroke:#333,stroke-width:2px
    classDef error fill:#fcc,stroke:#333,stroke-width:2px
    classDef antiPattern fill:#fcc,stroke:#f00,stroke-width:2px,stroke-dasharray: 5 5
    
    class PT100 singleton
    class PT101 factory
    class PT102 bridge
    class PT103 facade
    class PT104 callback
    class PT105 resource
    class PT106 error
    class AP015,AP016,AP017 antiPattern
```

## Desen Dağılımı (TreeMap)

```mermaid
graph TD
    subgraph SDK
        subgraph creation[Yaratıcı Desenler]
            PT100[PT-100: Singleton - 1 adet]
            PT101[PT-101: Factory Method - 2 adet]
        end
        
        subgraph structural[Yapısal Desenler]
            PT102[PT-102: Bridge - 8 adet]
            PT103[PT-103: Facade - 1 adet]
        end
        
        subgraph behavioral[Davranışsal Desenler]
            PT104[PT-104: Callback - 1 adet]
            PT105[PT-105: Resource Management - 4 adet]
        end
        
        subgraph error[Hata Yönetimi]
            PT106[PT-106: Error Handling - 3 adet]
        end
    end
    
    classDef creation fill:#ff9,stroke:#333,stroke-width:2px
    classDef structural fill:#9cf,stroke:#333,stroke-width:2px
    classDef behavioral fill:#9f9,stroke:#333,stroke-width:2px
    classDef error fill:#fcc,stroke:#333,stroke-width:2px
    
    class creation creation
    class structural structural
    class behavioral behavioral
    class error error
```

## Desen Etkinlik Grafiği

```
Etkinlik Skoru (0-1)
^
|
|                  * PT100 (0.95)
|                * PT103 (0.95)
|               * PT102 (0.94)
|              * PT105 (0.93)
|             * PT101 (0.92)
|             * PT106 (0.92)
|            * PT104 (0.90)
|
+----------------------------------------->
   Desen
```

## Desen Kullanım Haritası

```
Sınıf / Desen | PT-100 | PT-101 | PT-102 | PT-103 | PT-104 | PT-105 | PT-106
--------------|--------|--------|--------|--------|--------|--------|--------
M3TM          |   X    |        |        |   X    |        |   X    |        
M3TMModel     |        |        |   X    |        |        |   X    |        
M3TMModelMgr  |        |   X    |   X    |        |        |   X    |   X    
M3TMTrainingMgr|       |   X    |   X    |        |        |   X    |   X    
TrainingCB    |        |        |        |        |   X    |        |        
M3TMException |        |        |        |        |        |        |   X    
```

## Desen Olgunluk Isı Haritası

```
       PT-100    PT-101    PT-102    PT-103    PT-104    PT-105    PT-106
       ------    ------    ------    ------    ------    ------    ------
Etkk   █████     █████     █████     █████     ████▓     █████     █████
Tutr   █████     █████     █████     █████     █████     █████     █████
Bakm   █████     ████▓     ████▓     █████     ████▓     ████▓     █████
Genl   ████▓     █████     █████     ████▓     ████▓     ████▓     ████▓
Test   ████▓     █████     ████▓     ████▓     █████     █████     █████

█████ = Mükemmel (0.9-1.0)
████▓ = İyi (0.8-0.9)
███▓▓ = Orta (0.7-0.8)
██▓▓▓ = Geliştirilmeli (0.6-0.7)
█▓▓▓▓ = Zayıf (<0.6)

Etkk = Etkinlik
Tutr = Tutarlılık
Bakm = Bakım Kolaylığı
Genl = Genişletilebilirlik
Test = Test Edilebilirlik
```

## Desen Evrim Zaman Çizelgesi

```
2024-06-11 --------------------------------------------------------------------------
         |                                                                          
         |  PT-100, PT-101, PT-102, PT-103, PT-104, PT-105, PT-106                  
         |  (Android SDK Desenlerinin İlk Belgelendirilmesi)                       
         |                                                                          
Gelecek  --------------------------------------------------------------------------
         |                                                                          
         |  PT-104 -> Callback Deseninin Gelişimi                                   
         |  (Daha Granüler Callback'ler Eklenmesi)                                 
         |                                                                          
         |  PT-105 -> AutoCloseable İmplementasyonu                                 
         |  (Try-with-resources desteği)                                           
         |                                                                          
```

Bu görselleştirmeler, Android SDK desenlerinin ilişkilerini, dağılımını, etkinliğini ve evrimini anlamak için yardımcı olabilir. 
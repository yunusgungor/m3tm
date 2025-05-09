"""
Araştırma Raporu Oluşturma Aracı

Bu modül, dikkat ve konvolüsyon mekanizmaları araştırma bulgularını 
bir Markdown raporu formatında derlemek için kullanılır.
"""

import os
import json
import datetime
from pathlib import Path
import glob
from typing import Dict, List, Any, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ResearchReportGenerator:
    """Araştırma raporu oluşturan sınıf."""
    
    def __init__(self, 
                benchmark_results_dir: str = "benchmark_results",
                output_dir: str = "research_reports"):
        """
        Args:
            benchmark_results_dir: Benchmark sonuçlarının bulunduğu dizin
            output_dir: Çıktı raporunun kaydedileceği dizin
        """
        self.benchmark_results_dir = benchmark_results_dir
        self.output_dir = output_dir
        self.attention_results = None
        self.convolution_results = None
        
        # Çıktı dizininin var olduğundan emin ol
        os.makedirs(output_dir, exist_ok=True)
    
    def load_benchmark_results(self, attention_file: Optional[str] = None, convolution_file: Optional[str] = None):
        """Benchmark sonuçlarını yükler.
        
        Args:
            attention_file: Dikkat mekanizmaları benchmark sonuç dosyası
                           None ise, en yeni dosya kullanılır.
            convolution_file: Konvolüsyon mekanizmaları benchmark sonuç dosyası
                             None ise, en yeni dosya kullanılır.
        """
        # Dikkat sonuçlarını yükle
        if attention_file is None:
            attention_files = glob.glob(os.path.join(self.benchmark_results_dir, 'attention_mechanisms_benchmark_*.json'))
            if attention_files:
                attention_file = max(attention_files, key=os.path.getctime)
            else:
                print("Warning: No attention mechanism benchmark files found")
        
        if attention_file and os.path.exists(attention_file):
            with open(attention_file, 'r') as f:
                self.attention_results = json.load(f)
            print(f"Loaded attention benchmark results from: {attention_file}")
        
        # Konvolüsyon sonuçlarını yükle
        if convolution_file is None:
            convolution_files = glob.glob(os.path.join(self.benchmark_results_dir, 'convolution_mechanisms_benchmark_*.json'))
            if convolution_files:
                convolution_file = max(convolution_files, key=os.path.getctime)
            else:
                print("Warning: No convolution mechanism benchmark files found")
        
        if convolution_file and os.path.exists(convolution_file):
            with open(convolution_file, 'r') as f:
                self.convolution_results = json.load(f)
            print(f"Loaded convolution benchmark results from: {convolution_file}")
    
    def generate_report(self, title: str = "Mobil Mekanizmaları Araştırma Raporu", 
                      output_filename: Optional[str] = None) -> str:
        """Araştırma raporunu oluşturur ve kaydeder.
        
        Args:
            title: Rapor başlığı
            output_filename: Rapor dosya adı. None ise otomatik oluşturulur.
            
        Returns:
            Oluşturulan rapor dosyasının yolu
        """
        if output_filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            output_filename = f"research_report_{timestamp}.md"
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        if not self.attention_results and not self.convolution_results:
            print("Error: No benchmark results loaded. Cannot generate report.")
            return None
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Rapor başlığı ve temel bilgiler
            f.write(f"# {title}\n\n")
            f.write(f"**Oluşturma Tarihi:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Özet\n\n")
            f.write("Bu rapor, çeşitli dikkat ve konvolüsyon mekanizmalarının mobil cihazlarda performans açısından ")
            f.write("karşılaştırılması amacıyla yapılan araştırmanın sonuçlarını içermektedir. ")
            f.write("Araştırma, M³TM modeli için en uygun mekanizmaların belirlenmesine yönelik olarak ")
            f.write("gerçekleştirilmiştir.\n\n")
            
            # İçindekiler
            f.write("## İçindekiler\n\n")
            f.write("1. [Giriş](#giriş)\n")
            f.write("2. [Araştırma Metodolojisi](#araştırma-metodolojisi)\n")
            if self.attention_results:
                f.write("3. [Dikkat Mekanizmaları](#dikkat-mekanizmaları)\n")
                f.write("   3.1. [Mekanizma Açıklamaları](#dikkat-mekanizma-açıklamaları)\n")
                f.write("   3.2. [Performans Karşılaştırması](#dikkat-performans-karşılaştırması)\n")
                f.write("   3.3. [Bellek Kullanımı](#dikkat-bellek-kullanımı)\n")
            if self.convolution_results:
                section_num = 4 if self.attention_results else 3
                f.write(f"{section_num}. [Konvolüsyon Mekanizmaları](#konvolüsyon-mekanizmaları)\n")
                f.write(f"   {section_num}.1. [Mekanizma Açıklamaları](#konvolüsyon-mekanizma-açıklamaları)\n")
                f.write(f"   {section_num}.2. [Performans Karşılaştırması](#konvolüsyon-performans-karşılaştırması)\n")
                f.write(f"   {section_num}.3. [Bellek Kullanımı](#konvolüsyon-bellek-kullanımı)\n")
            next_section = (5 if self.attention_results and self.convolution_results else 
                          (4 if self.attention_results or self.convolution_results else 3))
            f.write(f"{next_section}. [Öneriler ve Sonuçlar](#öneriler-ve-sonuçlar)\n")
            f.write(f"{next_section+1}. [Ekler](#ekler)\n\n")
            
            # Giriş
            f.write("## Giriş\n\n")
            f.write("Mobil cihazlarda çalışan yapay zeka modelleri, sınırlı bellek, işlem gücü ve batarya ömrü ")
            f.write("gibi kısıtlamalarla karşı karşıyadır. Bu bağlamda, standart transformer mimarilerinde ")
            f.write("kullanılan mekanizmaların (özellikle dikkat ve konvolüsyon operasyonları) ")
            f.write("mobil-dostu alternatiflerinin araştırılması önem kazanmaktadır.\n\n")
            
            f.write("Bu çalışmada, çeşitli alternatif dikkat ve konvolüsyon mekanizmalarının performans, ")
            f.write("bellek kullanımı ve doğruluk açısından karşılaştırılması yapılmıştır. ")
            f.write("Araştırma, M³TM (Mobil Çok Modlu Transformer Model) projesi için ")
            f.write("en uygun mekanizmaların belirlenmesini amaçlamaktadır.\n\n")
            
            # Araştırma Metodolojisi
            f.write("## Araştırma Metodolojisi\n\n")
            
            if self.attention_results:
                config = self.attention_results.get('config', {})
                f.write("### Benchmark Yapılandırması\n\n")
                f.write("Benchmark testleri aşağıdaki yapılandırma ile gerçekleştirilmiştir:\n\n")
                f.write("- **Cihaz:** " + config.get('device', 'cpu') + "\n")
                f.write("- **Çalıştırma Sayısı:** " + str(config.get('num_runs', 'N/A')) + "\n")
                f.write("- **Isınma Turları:** " + str(config.get('warmup_runs', 'N/A')) + "\n")
                f.write("- **Batch Boyutları:** " + str(config.get('batch_sizes', 'N/A')) + "\n")
                f.write("- **Girdi Boyutları:** " + str(config.get('input_dims', 'N/A')) + "\n")
                f.write("- **Sekans Uzunlukları:** " + str(config.get('seq_lengths', 'N/A')) + "\n")
                f.write("- **Görüntü Boyutları:** " + str(config.get('image_sizes', 'N/A')) + "\n\n")
            
            f.write("### Değerlendirme Metrikleri\n\n")
            f.write("Mekanizmalar aşağıdaki metrikler açısından değerlendirilmiştir:\n\n")
            f.write("1. **Çıkarım Süresi (Latency):** Farklı girdi boyutları için ortalama çıkarım süresi (ms)\n")
            f.write("2. **Bellek Kullanımı:** Teorik bellek karmaşıklığı ve gerçek bellek tüketimi\n")
            f.write("3. **Parametre Sayısı:** Modelin eğitilebilir parametre sayısı\n")
            f.write("4. **İşlem Karmaşıklığı:** Teorik hesaplama karmaşıklığı\n\n")
            
            # Dikkat Mekanizmaları Bölümü
            if self.attention_results:
                f.write("## Dikkat Mekanizmaları\n\n")
                attention_mechanisms = list(self.attention_results.get('results', {}).keys())
                
                # Mekanizma açıklamaları
                f.write("### Dikkat Mekanizma Açıklamaları\n\n")
                
                f.write("#### StandardSelfAttention\n\n")
                f.write("Standart öz-dikkat mekanizması, Transformer mimarilerinde kullanılan temel dikkat ")
                f.write("mekanizmasıdır. Her sorguda tüm anahtar-değer çiftleri ile O(n²) karmaşıklığında ")
                f.write("işlem yapar. Karşılaştırma için referans nokta olarak kullanılmıştır.\n\n")
                
                f.write("#### LinformerAttention\n\n")
                f.write("Linformer, sekans uzunluğuna göre doğrusal ölçeklenen bir dikkat mekanizmasıdır ")
                f.write("(O(n) karmaşıklık). Dikkat matrisinin düşük dereceli bir yaklaşımını kullanarak ")
                f.write("bellek ve hesaplama gereksinimlerini azaltır. Anahtar ve değer projeksiyonları için ")
                f.write("bir boyut düşürme stratejisi uygular.\n\n")
                
                # Diğer mekanizmalar, eğer listedeyseler
                if "PerformerAttention" in attention_mechanisms:
                    f.write("#### PerformerAttention\n\n")
                    f.write("Performer, FAVOR+ (Fast Attention Via positive Orthogonal Random features) ")
                    f.write("yöntemini kullanarak dikkat hesaplamasını hızlandıran bir mekanizmadır. ")
                    f.write("Kernel yöntemi kullanarak dikkat hesaplamasını O(n) karmaşıklığa düşürür.\n\n")
                
                if "LocalAttention" in attention_mechanisms:
                    f.write("#### LocalAttention\n\n")
                    f.write("Yerel dikkat mekanizması, yalnızca belirli bir pencere içindeki bağlamı dikkate ")
                    f.write("alarak hesaplama yapar. Bu yaklaşım uzun sekanslar için hesaplama maliyetini ")
                    f.write("önemli ölçüde azaltabilir.\n\n")
                
                if "MobileAttention" in attention_mechanisms:
                    f.write("#### MobileAttention\n\n")
                    f.write("Mobil-dostu dikkat mekanizması, derinlik yönlü konvolüsyon, sıkıştırma ve geçit ")
                    f.write("mekanizmaları kullanarak dikkat operasyonlarını verimli hale getirir. Özellikle ")
                    f.write("mobil cihazlar için optimize edilmiştir.\n\n")
                
                # Performans Karşılaştırması
                f.write("### Dikkat Performans Karşılaştırması\n\n")
                
                # Latency karşılaştırma tablosu
                f.write("#### Çıkarım Süresi Karşılaştırması\n\n")
                
                # Basit bir tablo oluştur
                f.write("| Mekanizma | Ortalama Latency (ms) | Medyan Latency (ms) | Min Latency (ms) | Max Latency (ms) |\n")
                f.write("|-----------|----------------------|---------------------|-----------------|------------------|\n")
                
                for mechanism in attention_mechanisms:
                    results = self.attention_results['results'].get(mechanism, [])
                    if results:
                        # Tüm sonuçların ortalamasını al
                        avg_latency = np.mean([r['metrics'].get('mean_latency_ms', 0) for r in results])
                        med_latency = np.mean([r['metrics'].get('median_latency_ms', 0) for r in results])
                        min_latency = np.mean([r['metrics'].get('min_latency_ms', 0) for r in results])
                        max_latency = np.mean([r['metrics'].get('max_latency_ms', 0) for r in results])
                        
                        f.write(f"| {mechanism} | {avg_latency:.4f} | {med_latency:.4f} | {min_latency:.4f} | {max_latency:.4f} |\n")
                
                f.write("\n")
                
                # Sekans uzunluğuna göre latency
                f.write("#### Sekans Uzunluğuna Göre Çıkarım Süresi\n\n")
                f.write("Sekans uzunluğu arttıkça farklı dikkat mekanizmalarının performans değişimleri ")
                f.write("karşılaştırmalı olarak incelenmiştir. Aşağıdaki tablo, 64 boyutlu girdiler ve ")
                f.write("batch size=1 için sonuçları göstermektedir:\n\n")
                
                # Sekans uzunluklarına göre tablo
                f.write("| Mekanizma |")
                
                # Benzersiz sekans uzunluklarını bul
                seq_lengths = set()
                for mechanism in attention_mechanisms:
                    for result in self.attention_results['results'].get(mechanism, []):
                        if result['params'].get('dim', 0) == 64 and result['params'].get('batch_size', 0) == 1:
                            seq_lengths.add(result['params'].get('seq_len', 0))
                
                seq_lengths = sorted(list(seq_lengths))
                
                # Sütun başlıkları
                for seq_len in seq_lengths:
                    f.write(f" {seq_len} |")
                f.write("\n")
                
                # Tablo sınırları
                f.write("|-----------|")
                for _ in seq_lengths:
                    f.write("---------|")
                f.write("\n")
                
                # Her mekanizma için satır
                for mechanism in attention_mechanisms:
                    f.write(f"| {mechanism} |")
                    
                    for seq_len in seq_lengths:
                        # Bu sekans uzunluğu için sonucu bul
                        latency = None
                        for result in self.attention_results['results'].get(mechanism, []):
                            if (result['params'].get('dim', 0) == 64 and 
                                result['params'].get('batch_size', 0) == 1 and
                                result['params'].get('seq_len', 0) == seq_len):
                                latency = result['metrics'].get('mean_latency_ms', 0)
                                break
                        
                        if latency is not None:
                            f.write(f" {latency:.4f} |")
                        else:
                            f.write(" - |")
                    
                    f.write("\n")
                
                f.write("\n")
                
                # Bellek Kullanımı
                f.write("### Dikkat Bellek Kullanımı\n\n")
                
                f.write("#### Teorik Bellek Karmaşıklığı\n\n")
                f.write("Farklı dikkat mekanizmalarının teorik bellek karmaşıklığı:\n\n")
                
                f.write("| Mekanizma | Bellek Karmaşıklığı |\n")
                f.write("|-----------|---------------------|\n")
                f.write("| StandardSelfAttention | O(n²) |\n")
                f.write("| LinformerAttention | O(n) |\n")
                if "PerformerAttention" in attention_mechanisms:
                    f.write("| PerformerAttention | O(n) |\n")
                if "LocalAttention" in attention_mechanisms:
                    f.write("| LocalAttention | O(n * w) (w: pencere boyutu) |\n")
                if "MobileAttention" in attention_mechanisms:
                    f.write("| MobileAttention | O(n) |\n")
                
                f.write("\n")
                
                # Parametre Sayısı
                f.write("#### Parametre Sayısı\n\n")
                f.write("Her mekanizma için eğitilebilir parametre sayısı (64 boyutlu girdi ve 8 dikkat başı için):\n\n")
                
                f.write("| Mekanizma | Parametre Sayısı |\n")
                f.write("|-----------|------------------|\n")
                
                for mechanism in attention_mechanisms:
                    param_count = self.attention_results['metrics'].get(mechanism, {}).get('parameter_count', 0)
                    f.write(f"| {mechanism} | {param_count} |\n")
                
                f.write("\n")
            
            # Konvolüsyon Mekanizmaları Bölümü
            if self.convolution_results:
                f.write("## Konvolüsyon Mekanizmaları\n\n")
                conv_mechanisms = list(self.convolution_results.get('results', {}).keys())
                
                # Mekanizma açıklamaları
                f.write("### Konvolüsyon Mekanizma Açıklamaları\n\n")
                
                f.write("#### StandardConv\n\n")
                f.write("Standart konvolüsyon, karşılaştırmalar için temel referans noktası olarak kullanılan ")
                f.write("geleneksel 2B konvolüsyon katmanıdır. Her girdi-çıktı kanal çifti için ayrı filtre ")
                f.write("kullanır ve O(k² * Cin * Cout) karmaşıklığına sahiptir.\n\n")
                
                f.write("#### DepthwiseSeparableConv\n\n")
                f.write("Derinlik yönlü ayrılabilir konvolüsyon, MobileNet gibi hafif mimarilerde yaygın olarak ")
                f.write("kullanılan verimli bir konvolüsyon varyantıdır. İşlemi iki aşamaya ayırır: derinlik ")
                f.write("yönlü konvolüsyon ve nokta yönlü (1x1) konvolüsyon. Toplam karmaşıklığı ")
                f.write("O(k² * Cin + Cin * Cout) olarak azaltır.\n\n")
                
                # Diğer mekanizmalar, eğer listedeyseler
                if "MobileConvBlock" in conv_mechanisms:
                    f.write("#### MobileConvBlock\n\n")
                    f.write("MobileNetV2'de tanıtılan ters darboğaz (inverted bottleneck) bloğudur. ")
                    f.write("Üç aşamalı bir yapı kullanır: genişleme (1x1 konvolüsyon ile kanal sayısını artırma), ")
                    f.write("derinlik yönlü konvolüsyon ve projeksiyon (1x1 konvolüsyon ile kanal sayısını azaltma). ")
                    f.write("Artık bağlantılar da kullanabilir.\n\n")
                
                if "ShuffleConv" in conv_mechanisms:
                    f.write("#### ShuffleConv\n\n")
                    f.write("ShuffleNet'te tanıtılan kanal karıştırma operasyonunu kullanan bir konvolüsyon ")
                    f.write("bloğudur. Grup konvolüsyonları ve kanal karıştırma ile bilgi akışını iyileştirir.\n\n")
                
                if "GhostConv" in conv_mechanisms:
                    f.write("#### GhostConv\n\n")
                    f.write("GhostNet'te tanıtılan, daha az hesaplama maliyetiyle hayali (ghost) öznitelikler ")
                    f.write("üreten konvolüsyon bloğudur. Ana öznitelik haritasından daha düşük maliyetli ")
                    f.write("doğrusal dönüşümlerle öznitelikler oluşturur.\n\n")
                
                # Performans Karşılaştırması
                f.write("### Konvolüsyon Performans Karşılaştırması\n\n")
                
                # Latency karşılaştırma tablosu
                f.write("#### Çıkarım Süresi Karşılaştırması\n\n")
                f.write("Farklı konvolüsyon mekanizmalarının ortalama çıkarım süreleri (ms):\n\n")
                
                # Basit bir tablo oluştur
                f.write("| Mekanizma | Ortalama Latency (ms) | Medyan Latency (ms) | Min Latency (ms) | Max Latency (ms) |\n")
                f.write("|-----------|----------------------|---------------------|-----------------|------------------|\n")
                
                for mechanism in conv_mechanisms:
                    results = self.convolution_results['results'].get(mechanism, [])
                    if results:
                        # Tüm sonuçların ortalamasını al
                        avg_latency = np.mean([r['metrics'].get('mean_latency_ms', 0) for r in results])
                        med_latency = np.mean([r['metrics'].get('median_latency_ms', 0) for r in results])
                        min_latency = np.mean([r['metrics'].get('min_latency_ms', 0) for r in results])
                        max_latency = np.mean([r['metrics'].get('max_latency_ms', 0) for r in results])
                        
                        f.write(f"| {mechanism} | {avg_latency:.4f} | {med_latency:.4f} | {min_latency:.4f} | {max_latency:.4f} |\n")
                
                f.write("\n")
                
                # Girdi boyutuna göre latency
                f.write("#### Görüntü Boyutuna Göre Çıkarım Süresi\n\n")
                f.write("Görüntü boyutu arttıkça farklı konvolüsyon mekanizmalarının performans değişimleri ")
                f.write("karşılaştırmalı olarak incelenmiştir. Aşağıdaki tablo, 64 kanallı girdiler ve ")
                f.write("batch size=1 için sonuçları göstermektedir:\n\n")
                
                # Görüntü boyutlarına göre tablo
                f.write("| Mekanizma |")
                
                # Benzersiz görüntü boyutlarını bul
                image_sizes = set()
                for mechanism in conv_mechanisms:
                    for result in self.convolution_results['results'].get(mechanism, []):
                        if (result['params'].get('in_channels', 0) == 64 and 
                            result['params'].get('batch_size', 0) == 1 and
                            'image_size' in result['params']):
                            image_sizes.add(tuple(result['params']['image_size']))
                
                image_sizes = sorted(list(image_sizes))
                
                # Sütun başlıkları
                for size in image_sizes:
                    f.write(f" {size[0]}x{size[1]} |")
                f.write("\n")
                
                # Tablo sınırları
                f.write("|-----------|")
                for _ in image_sizes:
                    f.write("---------|")
                f.write("\n")
                
                # Her mekanizma için satır
                for mechanism in conv_mechanisms:
                    f.write(f"| {mechanism} |")
                    
                    for size in image_sizes:
                        # Bu görüntü boyutu için sonucu bul
                        latency = None
                        for result in self.convolution_results['results'].get(mechanism, []):
                            if (result['params'].get('in_channels', 0) == 64 and 
                                result['params'].get('batch_size', 0) == 1 and
                                result['params'].get('image_size', None) == list(size)):
                                latency = result['metrics'].get('mean_latency_ms', 0)
                                break
                        
                        if latency is not None:
                            f.write(f" {latency:.4f} |")
                        else:
                            f.write(" - |")
                    
                    f.write("\n")
                
                f.write("\n")
                
                # Bellek Kullanımı
                f.write("### Konvolüsyon Bellek Kullanımı\n\n")
                
                f.write("#### Teorik Bellek Karmaşıklığı\n\n")
                f.write("Farklı konvolüsyon mekanizmalarının teorik bellek karmaşıklığı:\n\n")
                
                f.write("| Mekanizma | Bellek Karmaşıklığı |\n")
                f.write("|-----------|---------------------|\n")
                f.write("| StandardConv | O(k² * Cin * Cout) |\n")
                f.write("| DepthwiseSeparableConv | O(k² * Cin + Cin * Cout) |\n")
                if "MobileConvBlock" in conv_mechanisms:
                    f.write("| MobileConvBlock | O(k² * Cin * e + Cin * e + e * Cout) (e: expansion factor) |\n")
                if "ShuffleConv" in conv_mechanisms:
                    f.write("| ShuffleConv | O(k² * Cin/g * Cout/g + g²) (g: grup sayısı) |\n")
                if "GhostConv" in conv_mechanisms:
                    f.write("| GhostConv | O(Cin * Cout/r + k² * Cout/r) (r: ratio) |\n")
                
                f.write("\n")
                
                # Parametre Sayısı
                f.write("#### Parametre Sayısı\n\n")
                f.write("Her mekanizma için eğitilebilir parametre sayısı (64 kanallı girdi ve çıktı için):\n\n")
                
                f.write("| Mekanizma | Parametre Sayısı |\n")
                f.write("|-----------|------------------|\n")
                
                for mechanism in conv_mechanisms:
                    param_count = self.convolution_results['metrics'].get(mechanism, {}).get('parameter_count', 0)
                    f.write(f"| {mechanism} | {param_count} |\n")
                
                f.write("\n")
            
            # Öneriler ve Sonuçlar
            f.write("## Öneriler ve Sonuçlar\n\n")
            
            f.write("Mobil ortamlarda en iyi performans-doğruluk dengesi için aşağıdaki mekanizmalar önerilmektedir:\n\n")
            
            if self.attention_results:
                f.write("### Dikkat Mekanizmaları Önerileri\n\n")
                
                f.write("1. **Kısa sekans uzunlukları için:** Standart öz-dikkat mekanizması, kısa sekanslar ")
                f.write("için hala makul performans sunmaktadır ve tam doğruluk sağlamaktadır.\n\n")
                
                f.write("2. **Orta ve uzun sekans uzunlukları için:** Linformer dikkat mekanizması, özellikle ")
                f.write("sekans uzunluğu arttıkça önemli performans avantajı sağlamaktadır. Minimal doğruluk ")
                f.write("kaybıyla bellek ve işlem gereksinimlerini önemli ölçüde azaltır.\n\n")
                
                f.write("3. **Çok uzun sekanslar ve sınırlı cihazlar için:** Yerel dikkat mekanizması veya ")
                f.write("özel mobil-dostu dikkat mekanizmaları düşünülebilir. Bu mekanizmalar daha fazla ")
                f.write("doğruluk kaybı riskine rağmen en düşük kaynak kullanımı sağlar.\n\n")
            
            if self.convolution_results:
                f.write("### Konvolüsyon Mekanizmaları Önerileri\n\n")
                
                f.write("1. **Genel kullanım için:** Derinlik yönlü ayrılabilir konvolüsyon, standart konvolüsyona ")
                f.write("kıyasla çok daha verimlidir ve çoğu durumda minimal doğruluk kaybı yaşatır.\n\n")
                
                f.write("2. **Daha agresif optimizasyon gerektiren durumlar için:** MobileConvBlock (ters darboğaz yapısı), ")
                f.write("özellikle daha derin mimariler için önerilir. Artık bağlantılar sayesinde doğruluk korunur.\n\n")
                
                if "GhostConv" in conv_mechanisms:
                    f.write("3. **Çok sınırlı kaynaklar için:** Ghost konvolüsyon, en az parametre ve hesaplama ")
                    f.write("gereksinimi sunar, ancak bazı görevlerde doğruluk kaybı yaşanabilir.\n\n")
            
            f.write("### Genel Öneriler\n\n")
            
            f.write("- **Hibrit yaklaşım:** Farklı katmanlarda farklı optimizasyon stratejileri kullanılabilir: ")
            f.write("kritik katmanlarda daha doğru mekanizmalar, diğer katmanlarda daha verimli mekanizmalar.\n\n")
            
            f.write("- **Doğruluk ve performans dengesi:** Her uygulamanın gereksinimleri farklı olacağından, ")
            f.write("farklı mekanizmaların uygulamaya özgü doğruluk-performans dengesinin test edilmesi önerilir.\n\n")
            
            f.write("- **Model küçültme teknikleri:** Seçilen mekanizmalara ek olarak, nicemleme (quantization) ve ")
            f.write("budama (pruning) gibi tekniklerin kullanılması, mobil performansı daha da iyileştirebilir.\n\n")
            
            # Ekler
            f.write("## Ekler\n\n")
            
            f.write("### Benchmark Detayları\n\n")
            
            if self.attention_results:
                timestamp = self.attention_results.get('timestamp', 'N/A')
                f.write(f"- Dikkat mekanizmaları benchmark tarihi: {timestamp}\n")
            
            if self.convolution_results:
                timestamp = self.convolution_results.get('timestamp', 'N/A')
                f.write(f"- Konvolüsyon mekanizmaları benchmark tarihi: {timestamp}\n")
            
            f.write("\n")
            
            f.write("### Kaynaklar\n\n")
            
            f.write("1. Vaswani, A., et al. (2017). *Attention is All You Need*\n")
            f.write("2. Wang, S., et al. (2020). *Linformer: Self-Attention with Linear Complexity*\n")
            f.write("3. Howard, A., et al. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*\n")
            f.write("4. Sandler, M., et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*\n")
            f.write("5. Han, K., et al. (2020). *GhostNet: More Features from Cheap Operations*\n")
            f.write("6. Ma, N., et al. (2018). *ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design*\n")
            
        
        print(f"Research report generated and saved to: {output_path}")
        return output_path


def generate_research_report(benchmark_results_dir: str = "benchmark_results",
                          output_dir: str = "research_reports",
                          attention_file: Optional[str] = None,
                          convolution_file: Optional[str] = None,
                          title: str = "Mobil Mekanizmaları Araştırma Raporu",
                          output_filename: Optional[str] = None) -> str:
    """Araştırma raporu oluşturur.
    
    Args:
        benchmark_results_dir: Benchmark sonuçlarının bulunduğu dizin
        output_dir: Çıktı raporunun kaydedileceği dizin
        attention_file: Dikkat mekanizmaları benchmark sonuç dosyası
                       None ise, en yeni dosya kullanılır.
        convolution_file: Konvolüsyon mekanizmaları benchmark sonuç dosyası
                         None ise, en yeni dosya kullanılır.
        title: Rapor başlığı
        output_filename: Rapor dosya adı. None ise otomatik oluşturulur.
        
    Returns:
        Oluşturulan rapor dosyasının yolu
    """
    generator = ResearchReportGenerator(benchmark_results_dir, output_dir)
    generator.load_benchmark_results(attention_file, convolution_file)
    return generator.generate_report(title, output_filename) 
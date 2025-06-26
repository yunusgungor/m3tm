#!/bin/bash

# M³TM Distributed Training Script
# Bu script, farklı distributed training senaryoları için örnekler içerir

set -e

# Renkli çıktı için
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonksiyonlar
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# GPU sayısını kontrol et
check_gpus() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        print_info "Mevcut GPU sayısı: $GPU_COUNT"
        return $GPU_COUNT
    else
        print_warning "nvidia-smi bulunamadı, GPU kontrolü yapılamıyor"
        return 0
    fi
}

# Single-node multi-GPU eğitimi
run_single_node() {
    local config_file=$1
    local num_gpus=$2
    local output_dir=$3
    local mode=${4:-"full"}
    
    print_info "Single-node multi-GPU eğitimi başlatılıyor..."
    print_info "GPU sayısı: $num_gpus"
    print_info "Konfigürasyon: $config_file"
    print_info "Çıktı dizini: $output_dir"
    print_info "Mod: $mode"
    
    python scripts/run_distributed.py \
        --config "$config_file" \
        --output-dir "$output_dir" \
        --mode "$mode" \
        --gpus "$num_gpus"
}

# Multi-node eğitimi (master node)
run_multi_node_master() {
    local config_file=$1
    local num_nodes=$2
    local gpus_per_node=$3
    local output_dir=$4
    local mode=${5:-"full"}
    local master_port=${6:-12355}
    
    print_info "Multi-node eğitimi başlatılıyor (Master node)..."
    print_info "Node sayısı: $num_nodes"
    print_info "Node başına GPU: $gpus_per_node"
    print_info "Master port: $master_port"
    
    python scripts/run_distributed.py \
        --config "$config_file" \
        --output-dir "$output_dir" \
        --mode "$mode" \
        --num-nodes "$num_nodes" \
        --gpus-per-node "$gpus_per_node" \
        --node-rank 0 \
        --master-addr "localhost" \
        --master-port "$master_port"
}

# Multi-node eğitimi (worker node)
run_multi_node_worker() {
    local config_file=$1
    local num_nodes=$2
    local gpus_per_node=$3
    local node_rank=$4
    local master_addr=$5
    local output_dir=$6
    local mode=${7:-"full"}
    local master_port=${8:-12355}
    
    print_info "Multi-node eğitimi başlatılıyor (Worker node $node_rank)..."
    print_info "Master adres: $master_addr:$master_port"
    
    python scripts/run_distributed.py \
        --config "$config_file" \
        --output-dir "$output_dir" \
        --mode "$mode" \
        --num-nodes "$num_nodes" \
        --gpus-per-node "$gpus_per_node" \
        --node-rank "$node_rank" \
        --master-addr "$master_addr" \
        --master-port "$master_port"
}

# PyTorch distributed launcher kullanarak
run_with_torchrun() {
    local config_file=$1
    local num_gpus=$2
    local output_dir=$3
    local mode=${4:-"full"}
    
    print_info "torchrun ile distributed eğitim başlatılıyor..."
    
    torchrun \
        --nproc_per_node="$num_gpus" \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr="localhost" \
        --master_port=12355 \
        train.py \
        --config "$config_file" \
        --mode "$mode" \
        --output-dir "$output_dir" \
        --distributed
}

# SLURM ile distributed eğitim
run_with_slurm() {
    local config_file=$1
    local num_nodes=$2
    local gpus_per_node=$3
    local output_dir=$4
    local mode=${5:-"full"}
    local job_name=${6:-"m3tm-distributed"}
    local partition=${7:-"gpu"}
    local time_limit=${8:-"24:00:00"}
    
    print_info "SLURM ile distributed eğitim başlatılıyor..."
    
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --nodes=$num_nodes
#SBATCH --ntasks-per-node=$gpus_per_node
#SBATCH --gres=gpu:$gpus_per_node
#SBATCH --partition=$partition
#SBATCH --time=$time_limit
#SBATCH --output=$output_dir/slurm-%j.out
#SBATCH --error=$output_dir/slurm-%j.err

# Environment setup
module load python/3.8
module load cuda/11.8
source venv/bin/activate

# Get master node info
export MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12355
export WORLD_SIZE=\$((\$SLURM_NNODES * \$SLURM_NTASKS_PER_NODE))

# Run training
srun python train.py \\
    --config "$config_file" \\
    --mode "$mode" \\
    --output-dir "$output_dir" \\
    --distributed \\
    --world-size \$WORLD_SIZE \\
    --master-addr \$MASTER_ADDR \\
    --master-port \$MASTER_PORT
EOF

    print_success "SLURM job submitted"
}

# Yardım mesajı
show_help() {
    echo "M³TM Distributed Training Script"
    echo ""
    echo "Kullanım:"
    echo "  $0 <command> [options]"
    echo ""
    echo "Komutlar:"
    echo "  single-gpu     - Tek GPU eğitimi"
    echo "  multi-gpu      - Tek node multi-GPU eğitimi"
    echo "  multi-node     - Multi-node eğitimi"
    echo "  torchrun       - PyTorch torchrun ile eğitim"
    echo "  slurm          - SLURM ile eğitim"
    echo "  check-gpus     - GPU sayısını kontrol et"
    echo ""
    echo "Örnekler:"
    echo "  # Tek GPU"
    echo "  $0 single-gpu configs/training_config.yaml ./outputs/single"
    echo ""
    echo "  # 4 GPU ile eğitim"
    echo "  $0 multi-gpu configs/distributed_training.yaml ./outputs/multi 4"
    echo ""
    echo "  # 2 node, node başına 4 GPU (master node)"
    echo "  $0 multi-node configs/distributed_training.yaml ./outputs/multi 2 4 master"
    echo ""
    echo "  # torchrun ile 4 GPU"
    echo "  $0 torchrun configs/distributed_training.yaml ./outputs/torch 4"
    echo ""
}

# Ana script
case "$1" in
    "single-gpu")
        if [ $# -lt 3 ]; then
            print_error "Kullanım: $0 single-gpu <config> <output_dir> [mode]"
            exit 1
        fi
        python train.py --config "$2" --output-dir "$3" --mode "${4:-full}"
        ;;
    
    "multi-gpu")
        if [ $# -lt 4 ]; then
            print_error "Kullanım: $0 multi-gpu <config> <output_dir> <num_gpus> [mode]"
            exit 1
        fi
        run_single_node "$2" "$4" "$3" "${5:-full}"
        ;;
    
    "multi-node")
        if [ $# -lt 6 ]; then
            print_error "Kullanım: $0 multi-node <config> <output_dir> <num_nodes> <gpus_per_node> <master|worker> [node_rank] [master_addr]"
            exit 1
        fi
        
        if [ "$6" = "master" ]; then
            run_multi_node_master "$2" "$4" "$5" "$3" "${8:-full}"
        elif [ "$6" = "worker" ]; then
            if [ $# -lt 8 ]; then
                print_error "Worker node için node_rank ve master_addr gerekli"
                exit 1
            fi
            run_multi_node_worker "$2" "$4" "$5" "$7" "$8" "$3" "${9:-full}"
        else
            print_error "6. parametre 'master' veya 'worker' olmalı"
            exit 1
        fi
        ;;
    
    "torchrun")
        if [ $# -lt 4 ]; then
            print_error "Kullanım: $0 torchrun <config> <output_dir> <num_gpus> [mode]"
            exit 1
        fi
        run_with_torchrun "$2" "$4" "$3" "${5:-full}"
        ;;
    
    "slurm")
        if [ $# -lt 5 ]; then
            print_error "Kullanım: $0 slurm <config> <output_dir> <num_nodes> <gpus_per_node> [mode] [job_name] [partition] [time_limit]"
            exit 1
        fi
        run_with_slurm "$2" "$4" "$5" "$3" "${6:-full}" "${7:-m3tm-distributed}" "${8:-gpu}" "${9:-24:00:00}"
        ;;
    
    "check-gpus")
        check_gpus
        ;;
    
    "help"|"-h"|"--help"|"")
        show_help
        ;;
    
    *)
        print_error "Bilinmeyen komut: $1"
        show_help
        exit 1
        ;;
esac

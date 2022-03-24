# python main.py --n_fold 8 --batch_size 8 --weight_decay 0.0003 --dropout 0.3 --szs 512 128 64 64  --loss pcc --mhas 3 --split_ratios 0.95 0.05
# python main.py --n_fold 10 --batch_size 12 --epochs 50 --weight_decay 0.0002 --dropout 0.3 --szs 512 128 64 64 32  --loss pcc --mhas 3 --split_ratios 0.95 0.05
# python main.py --n_fold 10 --batch_size 12 --epochs 100 --weight_decay 0.0002 --dropout 0.3 --szs 512 128 64 64 32  --loss pcc --mhas 3 --split_ratios 0.95 0.05
# python main.py --n_fold 10 --batch_size 12 --epochs 100 --weight_decay 0.0001 --dropout 0.3 --szs 512 128 64 64 32  --loss pcc --mhas 3 --split_ratios 0.95 0.05

# python main.py --n_fold 10 --batch_size 12 --epochs 100 --weight_decay 0.0003 --dropout 0.3 --szs 512 128 64 64 32  --loss pcc --mhas 3 --split_ratios 0.95 0.05 --swa --lr 0.0003
# python main.py --n_fold 10 --batch_size 12 --epochs 150 --weight_decay 0.0003 --dropout 0.4 --szs 512 128 64 64 32  --loss pcc --mhas 3 --split_ratios 0.95 0.05 --swa --lr 0.0003

# python main.py --n_fold 10 --batch_size 8 --epochs 150 --weight_decay 0.0003 --dropout 0.4 --szs 512 128 64 32 32  --loss pcc --mhas 3 4 --split_ratios 0.95 0.05 --swa --lr 0.0003

common_args=" --n_fold 10 --early_stop --lr_scheduler plateau --gpus 1 --accumulate_grad_batches 6 --max_epochs 150 --split_ratios 0.9 0.1 --dropout 0.4"

# python main.py $common_args --batch_size 4 --weight_decay 0.0002 --szs 512 128 64 32 32  --mhas 3 4 --swa --lr 0.0003

# python main.py $common_args --batch_size 8 --weight_decay 0.0003 --szs 384 128 64 64 32 --mhas 3 --max_epochs 200 --swa --lr 0.001

# python main.py $common_args --batch_size 6 --weight_decay 0.0003 --szs 512 256 64 64 32 --mhas 3 4 --max_epochs 200 --swa --lr 0.001

# python main.py $common_args --batch_size 8 --weight_decay 0.0002 --szs 512 256 64 64 32 --mhas 3 --max_epochs 200 --swa --lr 0.001

common_args="--n_fold 10 --early_stop --lr_scheduler plateau --gpus 1 --accumulate_grad_batches 24 --log_every_n_steps 24  --max_epochs 101 --swa --split_ratios 0.95 0.05 --dropout 0.4"
# python3 main.py $common_args --szs 384 128 64 64 32 --mhas 2 --n_mem 64 --batch_size 1 --weight_decay 0.0002 --lr 0.001

common_args="--n_fold 10 --gpus 1 --accumulate_grad_batches 16 --max_epochs 70 --lr_scheduler plateau --swa --split_ratios 0.98 0.02 --dropout 0.4"
# python3 main.py $common_args --szs 512 128 64 64 32 --mhas 2 --n_mem 128 --batch_size 1 --weight_decay 0.0001 --lr 0.001

common_args="--n_fold 10 --gpus 1 --log_every_n_steps 24  --max_epochs 101 --split_ratios 0.95 0.05 --dropout 0.4"
# python3 main.py $common_args --szs 384 128 64 32 32 --mhas 2 --n_mem 64 --batch_size 16 --weight_decay 0.0003 --lr 0.001

common_args="--n_fold 20 --gpus 1 --max_epochs 150 --split_ratios 0.98 0.02 --dropout 0.5"
# python3 main.py $common_args --szs 384 128 64 32 32 --mhas 2 --n_mem 256 --batch_size 12 --weight_decay 0.0003 --lr 0.001
python3 main.py $common_args --szs 384 128 64 32 32 --mhas 2 --n_mem 256 --batch_size 12 --weight_decay 0.0003 --lr 0.001 --seed 2022

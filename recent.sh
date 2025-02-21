

source /home86/yunxshi/.bashrc

RATIO_LIST=(0.005)

for RATIO in "${RATIO_LIST[@]}"
do
    python CBS.py \
        --sampling 'recent' \
        --persona_learning_type 'distill' \
        --dataset 'Books' \
        --subset '480' \
        --ratio $RATIO
done


wait  # 等待所有后台进程完成
echo "All done!"

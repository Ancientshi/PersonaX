# parser.add_argument('--distance_threshold', type=float, default=0.5, help='distance_threshold')
# parser.add_argument('--alpha', type=float, default=1.06, help='alpha')
# parser.add_argument('--ratio', type=float, default=0.6, help='ratio')

source /home86/yunxshi/.bashrc

RATIO_LIST=(0.005)
for RATIO in "${RATIO_LIST[@]}"
do
    python CBS.py \
        --sampling 'relevance' \
        --persona_learning_type 'distill' \
        --dataset 'Books' \
        --subset '480' \
        --ratio $RATIO
done

wait  # 等待所有后台进程完成
echo "All done!"

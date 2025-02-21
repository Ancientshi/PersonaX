

source /home86/yunxshi/.bashrc


SIGIR_KEY=''
DEV_KEY=''



## Summarization in Books_480
RATIO_LIST=(0.5)
ALPHA_LIST=(1.06)
SAMPLING_LIST=('CBS_relevance')
DISTANCE_LIST=(0.7)
MAX_JOBS=2  # 最大并行进程数
CURRENT_JOBS=0

for RATIO in "${RATIO_LIST[@]}"
do
    for SAMPLING in "${SAMPLING_LIST[@]}"
    do
        for ALPHA in "${ALPHA_LIST[@]}"
        do
            for DISTANCE in "${DISTANCE_LIST[@]}"
            do 
                python CBS_relevance.py \
                    --api_key $SIGIR_KEY \
                    --dataset 'Books' \
                    --persona_learning_type 'distill' \
                    --sampling $SAMPLING \
                    --subset '480' \
                    --distance_threshold $DISTANCE \
                    --alpha $ALPHA \
                    --ratio $RATIO   # 后台运行任务
                ((CURRENT_JOBS+=1))
                
                # 检查并等待进程数量是否达到限制
                if [[ $CURRENT_JOBS -ge $MAX_JOBS ]]; then
                    wait -n  # 等待最早完成的一个后台任务
                    ((CURRENT_JOBS-=1))
                fi
            done
        done
    done
done

# 等待所有剩余的后台任务完成
wait
echo "All done!"




#### for Reflection on CDs_50
RATIO_LIST=(0.5 0.6 0.7 0.8 0.9 0.99)
SAMPLING_LIST=('CBS_relevance')

for RATIO in "${RATIO_LIST[@]}"
do
    for SAMPLING in "${SAMPLING_LIST[@]}"
    do
        python CBS_relevance.py \
            --api_key $SIGIR_KEY \
            --dataset 'CDs_and_Vinyl' \
            --persona_learning_type 'pairwise' \
            --sampling $SAMPLING \
            --subset '50' \
            --distance_threshold 0.7 \
            --alpha 1.01 \
            --ratio $RATIO   # 后台运行任务
    done
done
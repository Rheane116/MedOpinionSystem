#!/bin/bash

# 脚本名称: run_main_with_keywords.sh
# 功能: 循环运行main.py，依次传入keywords参数的不同值
# 用法: ./run_main_with_keywords.sh

# 定义关键词列表
KEYWORDS=("重庆医科大学附属第一医院" "四川大学华西口腔医院")

# 存放失败关键词的 JSON 文件
FAILED_JSON="add_scripts/add_log/dy_dy_failed_keywords.json"
FAILED_KEYWORDS=()


# 将脚本的echo输出重定向到add_scripts/add_log/zhihu.log，同时保持Python脚本的日志输出到./logs/zhihu目录
{
    echo "开始运行 main.py"
    echo "关键词列表: ${KEYWORDS[*]}"
    echo "========================================"

    # 循环运行main.py，依次传入每个关键词
    for keyword in "${KEYWORDS[@]}"; do
        echo "运行: python main.py --keywords \"$keyword\""
        echo "----------------------------------------"
        
        # 运行Python脚本，保持其日志输出到./logs/zhihu目录
        python main.py --keywords "$keyword" \
        --platform "dy" \
        --max_note 100 \
        --max_comment 100 \
        
        
        # 检查运行状态
        if [ $? -eq 0 ]; then
            echo "成功: 关键词 '$keyword' 运行完成"
        else
            echo "失败: 关键词 '$keyword' 运行失败"
            FAILED_KEYWORDS+=("$keyword")
        fi
        
        echo "----------------------------------------"
        echo
        
        # 等待1秒再运行下一个（可选）
        sleep 1
    done

    echo "所有关键词运行完成!"

    # 把失败关键词写入 JSON 文件
    if [ ${#FAILED_KEYWORDS[@]} -gt 0 ]; then
        printf "[\n" > "$FAILED_JSON"
        for i in "${!FAILED_KEYWORDS[@]}"; do
            if [ $i -lt $((${#FAILED_KEYWORDS[@]} - 1)) ]; then
                printf "  \"%s\",\n" "${FAILED_KEYWORDS[$i]}" >> "$FAILED_JSON"
            else
                printf "  \"%s\"\n" "${FAILED_KEYWORDS[$i]}" >> "$FAILED_JSON"
            fi
        done
        printf "]\n" >> "$FAILED_JSON"
        echo "失败关键词已写入 $FAILED_JSON"
    else
        echo "所有关键词均运行成功，无需生成失败关键词 JSON 文件"
    fi
} 2>&1 | tee -a add_scripts/add_log/dy.log
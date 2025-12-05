#!/bin/bash

# 脚本名称: run_main_with_keywords.sh
# 功能: 循环运行main.py，依次传入keywords参数的不同值
# 用法: ./run_main_with_keywords.sh

# 定义关键词列表
KEYWORDS=("首都医科大学附属北京同仁医院", "上海市第六人民医院", "上海交通大学医学院附属第九人民医院", "复旦大学附属儿科医院", "复旦大学附属肿瘤医院", "海军军医大学第一附属医院", "江苏省人民医院（南京医科大学第一附属医院）", "南京大学医学院附属鼓楼医院", "山东大学齐鲁医院", "广东省人民医院", "广州医科大学附属第一医院", "中山大学肿瘤防治中心", "陆军军医大学第一附属医院", "四川省人民医院", "北京积水潭医院", "首都医科大学附属北京安贞医院", "首都医科大学宣武医院", "中国医科大学附属盛京医院", "上海市肺科医院", "上海交通大学医学院附属新华医院", "复旦大学附属眼耳鼻喉科医院", "中国人民解放军东部战区总医院", "东南大学附属中大医院", "苏州大学附属第一医院", "浙江大学医学院附属邵逸夫医院", "福建医科大学附属第一医院", "南昌大学第一附属医院", "山东第一医科大学附属省立医院（山东省立医院）", "青岛大学附属医院", "武汉大学人民医院", "武汉大学中南医院", "中山大学附属第三医院", "重庆医科大学附属第一医院", "四川大学华西口腔医院")

# 存放失败关键词的 JSON 文件
FAILED_JSON="add_scripts/add_log/wb_failed_keywords.json"
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
        --platform "wb" \
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
} 2>&1 | tee add_scripts/add_log/wb.log
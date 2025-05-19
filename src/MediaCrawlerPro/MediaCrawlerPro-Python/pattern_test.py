import re
patterns = []

def set_pattern(agent_name):
    # 定义正则表达式模式
    formats = [
        re.compile(r'(.+)省(.*)(医院|中医院|妇幼保健院|中心)'),
        re.compile(r'(.+)市(.+)区(.*)(医院|中医院|妇幼保健院|中心)'),  # xx市xx区xx医院
        re.compile(r'(.+)市(.*)(医院|中医院|妇幼保健院|中心)'),
        re.compile(r'(.+大学|.+学院)附属第([一二三四五六七八九])(.*)(医院|中心)'),  # xx大学附属xx医院
        re.compile(r'(.+大学|.+学院)第([一二三四五六七八九])附属(.*)(医院|中心)'),
        re.compile(r'(.+大学|.+学院)(.*)(医院|中心)'),
        re.compile(r'(.*)(医院|中医院|妇幼保健院|中心)')
    ]
    if formats[0].search(agent_name):
        None

    # 定义一个替换函数
    def replace_match(match):
        if match.re.pattern == patterns[0].pattern:
            # 匹配 xx市xx区人民医院
            return match.group(1) + match.group(2) + '人民医院'
        elif match.re.pattern == patterns[1].pattern:
            # 匹配 xx大学附属xx医院
            return match.group(2)

    # 遍历模式并替换文本中的匹配项
    for pattern in patterns:
        text = pattern.sub(lambda m: replace_match(m), text)

    # 提取包含 "xxxx人民医院" 和 "xx医院" 的部分（假设 "xxxx人民医院" 和 "xx医院" 之间可能有其他文字）
    final_pattern = re.compile(r'.*?((?:\w+人民医院)|(?:\w+医院)).*?')
    matches = final_pattern.findall(text)

    # 过滤并去重，提取需要的部分
    filtered_results = set(match[0] for match in matches if match)

    return list(filtered_results)

if __name__ == '__main__':
    agent_name = input("输入医疗机构名：")
    formats = [
        re.compile(r'(.+)省(.*)(医院|中医院|保健院|中心)'),
        re.compile(r'(.+)市(.+)区(.*)(医院|中医院|保健院|中心)'),  # xx市xx区xx医院
        re.compile(r'(.+)市(.*)(医院|中医院|保健院|中心)'),
        re.compile(r'(.+大学|.+学院)附属第([一二三四五六七八九])(.*)(医院|中心)'),  # xx大学附属xx医院
        re.compile(r'(.+大学|.+学院)第([一二三四五六七八九])附属(.*)(医院|中心)'),
        re.compile(r'(.+大学|.+学院)(.*)(医院|中心)'),
        re.compile(r'(.*)科(医院|中医院|保健院|中心)'),
        re.compile(r'(.*)(医院|中医院|保健院|中心)')
    ]
    for i in range(len(formats)):
        match = formats[i].search(agent_name)
        if match:
            print(f"\n这是第{i}个format:\n")
            groups = match.groups()
            #for j in range(len(groups) + 1):
            #    print(match.group(j))
            if i == 0:
                patterns.append(re.compile(rf'省?{re.escape(groups(2))}({re.escape(groups(3))})?'))
                print(patterns[0])
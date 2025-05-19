import re






'''import re

text = "The date today is 2023-56-01, let's remember it. 2024-11-08 is gone."

# 使用search()方法在整个字符串中搜索日期格式
pattern = re.compile(r'(0|1?\d\d\d|20(?:[01]\d|(2[0-4])))-(0[1-9]|1[12])-(0[1-9]|[12]\d|30)')
search_result = pattern.finditer(text)
for match in search_result:
    print(match.group(0))
    print(match.group(1))
    print(match.group(2))
    print(match.group(3))
    print(match.group(4))
    print(match.group(5))
    

text = "colors: red, colors:blue; shapes: square, shapes:circle"

# 匹配颜色或形状
pattern = re.compile(r'(z?:colors?[:\s]+(\w+)(?:[,;\s]|$))|(?:shapes?[:\s]+(\w+)(?:[,;\s]|$))')

for match in pattern.finditer(text):
    if match.group(1):  # 如果是颜色
        print(f"Color found: {match.group(1)}")
    elif match.group(2):  # 如果是形状
        print(f"Shape found: {match.group(2)}")'''


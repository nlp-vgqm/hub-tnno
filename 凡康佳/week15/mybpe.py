with open("tokenizer_bpe.ipynb", "r",encoding='utf-8') as f:
    files=f.read()


files_utf8 = list(files.encode('utf-8'))

def calcalate_max_frequency(files_utf8):
    maps = {}
    for m,n in zip(files_utf8, files_utf8[1:]):
        if (m,n) in maps:
            maps[(m,n)] += 1
        else:
            maps[(m,n)] = 1

    max_frequency = 0
    max_tuple = ()

    for key, value in maps.items():
        if value > max_frequency:
            max_frequency = value
            max_tuple = key

    return max_tuple


def merge(token, max_tuple,new_token):
    for i in range(len(token)-1):
        # print(token[i:i+2])
        # print(list(max_tuple))
        if token[i:i+2] == list(max_tuple):
            token[i] = new_token
            del token[i+1]
    return token
    
new_token = 256
list_token=[]
for i in range(20):
    max_tuple = calcalate_max_frequency(files_utf8)
    new_token = new_token+1
    print(f'============={i+1}次压缩==============')
    # print(max_tuple)
    # print(new_token)
    origin_len = len(files_utf8)
    # print(len(files_utf8))
    files_utf8 = merge(files_utf8, max_tuple, new_token)
    # print(len(files_utf8))
    print(f'压缩率{(len(files_utf8))/origin_len:.2%}')

    list_token.append((new_token,max_tuple))

def decode(token, list_token):
    # 构建映射：新 token -> (x, y)
    mapping = {new_id: pair for new_id, pair in list_token}
    
    # 从最大的 token 开始替换（确保高层先展开）
    # 但更简单的方式是：只要还有 >=256 的 token，就继续替换
    
    result = token[:]  # 复制，避免修改原列表
    
    while any(t >= 256 for t in result):
        new_result = []
        for t in result:
            if t >= 256 and t in mapping:
                # 展开为两个元素
                new_result.extend(mapping[t])
            else:
                new_result.append(t)
        result = new_result
    
    return result


decode_token = decode(files_utf8,list_token)
print(max(decode_token))

decode_files = bytes(decode_token).decode('utf-8')

with open("decode_token.ipynb", "w",encoding='utf-8') as f:
    f.write(decode_files)

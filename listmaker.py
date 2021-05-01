import os

def make_list(root_dir, dsname):
    path_list = []
    src_dir = os.path.join(root_dir, dsname)
    for root, _, files in os.walk(src_dir):
        if not files:
            continue
        
        for file in files:
            name, ext = os.path.splitext(file)
            if ext == '.webm':
                path = os.path.join(root_dir, file)
                path_list.append(path + '\n')
    
    file_path = os.path.join(root_dir, f'{dsname}list.txt')
    with open(file_path, 'w') as f:
        f.writelines(path_list)

make_list('D:\SUTD/Term-7\Deep_Learning\BigProject\Action_Detection_In_Videos\data', 'train')
make_list('D:\SUTD/Term-7\Deep_Learning\BigProject\Action_Detection_In_Videos\data', 'test')
import subprocess

def check_tmp_filesystem():
    """检查 /tmp 是否为内存文件系统"""
    try:
        result = subprocess.run(['df', '-T', '/tmp'], 
                              capture_output=True, text=True)
        if 'tmpfs' in result.stdout:
            print("/tmp 使用 tmpfs (内存文件系统) - 极快的IO性能!")
        else:
            print("/tmp 使用常规磁盘文件系统")
        print(result.stdout)
    except:
        pass

check_tmp_filesystem()